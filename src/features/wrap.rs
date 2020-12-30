use std::marker::PhantomData;

use syntect::highlighting::Style as SyntectStyle;
use unicode_segmentation::UnicodeSegmentation;

use crate::config::Config;
use crate::delta::State;
use crate::features::line_numbers;
use crate::features::side_by_side::Panel;
use crate::features::OptionValueFunction;
use crate::style::Style;

pub fn make_feature() -> Vec<(String, OptionValueFunction)> {
    builtin_feature!([
        (
            "side-by-side-wrapped",
            bool,
            None,
            _opt => true
        ),
        (
            "side-by-side",
            bool,
            None,
            _opt => true
        )
    ])
}

#[derive(Debug, Clone)]
pub struct PlusMinus<P, T> {
    pub minus: T,
    pub plus: T,
    _marker: PhantomData<P>,
}

impl<P, T> PlusMinus<P, T> {
    pub fn new(minus: T, plus: T) -> Self {
        PlusMinus {
            minus,
            plus,
            _marker: PhantomData,
        }
    }
}

impl<P, T: Default> Default for PlusMinus<P, T> {
    fn default() -> Self {
        PlusMinus {
            minus: T::default(),
            plus: T::default(),
            _marker: PhantomData,
        }
    }
}

type LineWidthPM = PlusMinus<(), usize>;

pub fn available_line_width(
    config: &Config,
    line_numbers_data: &line_numbers::LineNumbersData,
) -> LineWidthPM {
    let (left_width, right_width) = line_numbers_data.formatted_width();

    // The width can be reduced by the line numbers and/or a possibly kept 1-wide "+/-/ " prefix.
    let mk_line_width = |panel: &Panel, panel_width| {
        panel
            .width
            .saturating_sub(panel_width)
            .saturating_sub(config.keep_plus_minus_markers as usize)
    };

    LineWidthPM::new(
        mk_line_width(&config.side_by_side_data.left_panel, left_width),
        mk_line_width(&config.side_by_side_data.right_panel, right_width),
    )
}

fn might_wrap<S>(line: &[(S, &str)], line_width: usize) -> bool
where
    S: Copy + Default + std::fmt::Debug,
{
    let line_sum = line.iter().fold(0, |sum, line_part| {
        sum + line_part.1.graphemes(true).count()
    });

    // + safety factor, better to underestimate. Could be the +/- prefix or the
    // trailing newline which does not count.
    line_sum + 3 > line_width
}

// Given a list of input lines, return whether any is too long, and a more specific
// list indicating which indiviual line is too long.
fn is_too_long<S>(line: &[Vec<(S, &str)>], line_width: usize) -> (bool, Vec<bool>)
where
    S: Copy + Default + std::fmt::Debug,
{
    let mut wrap_any = false;
    let wrapping_lines = line
        .iter()
        .map(|line| might_wrap(line, line_width))
        .inspect(|b| wrap_any |= b)
        .collect();

    (wrap_any, wrapping_lines)
}

pub fn minus_plus_too_long<'a, S>(
    minus_line: &[Vec<(S, &'a str)>],
    plus_line: &[Vec<(S, &'a str)>],
    line_width: &LineWidthPM,
) -> (bool, PlusMinus<S, Vec<bool>>)
where
    S: Copy + Default + std::fmt::Debug,
{
    let (wrap_minus, minus_wrapping_lines) = is_too_long(&minus_line, line_width.minus);
    let (wrap_plus, plus_wrapping_lines) = is_too_long(&plus_line, line_width.plus);

    (
        wrap_minus || wrap_plus,
        PlusMinus::new(minus_wrapping_lines, plus_wrapping_lines),
    )
}

// Wrap the given `line` if it is longer than `line_width`. Place `wrap_symbol` in the
// last position of the wrapped line. Wrap to at most `max_lines` lines.
//
// The input `line` is expected to start with an (ultimately not printed) "+/-/ " prefix.
// A prefix ("_") is also added to the start of wrapped lines.
pub fn wrap_line<'a, I, S>(
    line: I,
    wrap_symbol: &'a str,
    line_width: usize,
    max_lines: usize,
) -> Vec<Vec<(S, &'a str)>>
where
    I: IntoIterator<Item = (S, &'a str)>,
    <I as IntoIterator>::IntoIter: DoubleEndedIterator,
    S: Copy + Default + std::fmt::Debug,
{
    let mut result = vec![];

    // Symbol which:
    //  - represents the additional "+/-/ " prefix on the unwrapped input line, its
    //    length is added to the line_width.
    //  - is added to the beginning of wrapped lines so the wrapped lines also have
    //    a prefix (which is not printed).
    const LINEPREFIX: &str = "_";
    static_assertions::const_assert_eq!(LINEPREFIX.len(), 1); // must be a 1-byte char

    let max_len = line_width + LINEPREFIX.len();

    // Just in case: guard against infinite loops.
    let mut n = max_len * max_lines * 2;

    let mut curr_line = vec![];
    let mut curr_sum = 0;

    let mut stack = line.into_iter().rev().collect::<Vec<_>>();

    while !stack.is_empty() && result.len() + 1 < max_lines && max_len > LINEPREFIX.len() && n > 0 {
        n -= 1;

        let (style, text, graphemes) = stack
            .pop()
            .map(|(style, text)| (style, text, text.grapheme_indices(true).collect::<Vec<_>>()))
            .unwrap();
        let new_sum = curr_sum + graphemes.len();

        let must_split = if new_sum < max_len {
            curr_line.push((style, text));
            curr_sum = new_sum;
            false
        } else if new_sum == max_len {
            match stack.last() {
                // Perfect fit, no need to make space for a `wrap_symbol`.
                None => {
                    curr_line.push((style, text));
                    false
                }
                // A single '\n' left on the stack can be pushed onto the current line.
                Some((next_style, nl)) if stack.len() == 1 && *nl == "\n" => {
                    curr_line.push((style, text));
                    curr_line.push((*next_style, *nl));
                    stack.pop();
                    false
                }
                _ => true,
            }
        } else if new_sum == max_len + 1 && stack.is_empty() {
            // If the one overhanging char is '\n' then keep it on the current line.
            if !text.is_empty() && *text.as_bytes().last().unwrap() == b'\n' {
                curr_line.push((style, text));
                false
            } else {
                true
            }
        } else {
            true
        };

        // Text must be split, one part (or just `wrap_symbol`) is added to the
        // current line, the other is pushed onto the stack.
        if must_split {
            let grapheme_split_pos = graphemes.len() - (new_sum - max_len) - 1;

            let next_line = if grapheme_split_pos == 0 {
                text
            } else {
                let byte_split_pos = graphemes[grapheme_split_pos].0;
                let this_line = &text[..byte_split_pos];
                curr_line.push((style, this_line));
                &text[byte_split_pos..]
            };
            stack.push((style, next_line));

            curr_line.push((S::default(), &wrap_symbol));
            result.push(curr_line);

            curr_line = vec![(S::default(), LINEPREFIX)];
            curr_sum = LINEPREFIX.len();
        }
    }

    if !curr_line.is_empty() || result.is_empty() {
        result.push(curr_line);
    }

    if !stack.is_empty() {
        // `unwrap()` is ok, the previous `if` ensured `result` is not empty
        result.last_mut().unwrap().extend(stack.into_iter().rev());
    }

    result
}

pub fn wrap_text_and_style<'a, S>(
    folded_vec: &mut Vec<Vec<(S, &'a str)>>,
    input_vec: Vec<(S, &'a str)>,
    must_wrap: bool,
    wrap_symbol: &'a str,
    line_width: usize,
    max_lines: usize,
) -> (usize, usize)
where
    S: Copy + Default + std::fmt::Debug,
{
    let size_prev = folded_vec.len();

    if must_wrap {
        folded_vec.append(&mut wrap_line(
            input_vec.into_iter(),
            &wrap_symbol,
            line_width,
            max_lines,
        ));
    } else {
        folded_vec.push(input_vec.to_vec());
    }

    (size_prev, folded_vec.len())
}

#[allow(clippy::comparison_chain, clippy::type_complexity)]
pub fn wrap_plusminus_block<'c: 'a, 'a, S>(
    config: &'c Config,
    minus: Vec<Vec<(S, &'a str)>>,
    plus: Vec<Vec<(S, &'a str)>>,
    alignment: &[(Option<usize>, Option<usize>)],
    line_width: &LineWidthPM,
    wrapinfo: PlusMinus<S, Vec<bool>>,
) -> (
    Vec<(Option<usize>, Option<usize>)>,
    PlusMinus<(), Vec<State>>,
    PlusMinus<S, Vec<Vec<(S, &'a str)>>>,
)
where
    S: Copy + Default + std::fmt::Debug,
{
    let mut new_alignment = vec![];

    let mut new_minus_state = vec![];
    let mut new_plus_state = vec![];

    let mut new_minus = vec![];
    let mut new_plus = vec![];

    let mut minus_iter = minus.into_iter();
    let mut plus_iter = plus.into_iter();

    let mut minus_must_wrap_iter = wrapinfo.minus.into_iter();
    let mut plus_must_wrap_iter = wrapinfo.plus.into_iter();

    macro_rules! assert_alignment {
        ($have:tt, $expected:tt, $msg2:tt) => {
            assert_eq!(*$have, $expected, "bad alignment index {}", $msg2);
            $expected += 1;
        };
    }
    let mut m_expected = 0;
    let mut p_expected = 0;

    // Process blocks according to alignment and build a new alignment.
    // Wrapped lines get assigned the state HunkMinusWrapped/HunkPlusWrapped.
    for (minus, plus) in alignment {
        let (minus_extended, plus_extended) = match (minus, plus) {
            (Some(m), None) => {
                assert_alignment!(m, m_expected, "l (-)");

                let (minus_start, extended_to) = wrap_text_and_style(
                    &mut new_minus,
                    minus_iter.next().expect("bad alignment l (-)"),
                    minus_must_wrap_iter.next().expect("bad wrap info l (-)"),
                    &config.wrap_symbol,
                    line_width.minus,
                    config.wrap_max_lines,
                );

                for i in minus_start..extended_to {
                    new_alignment.push((Some(i), None));
                }

                (extended_to - minus_start, 0)
            }
            (None, Some(p)) => {
                assert_alignment!(p, p_expected, "(-) r");

                let (plus_start, extended_to) = wrap_text_and_style(
                    &mut new_plus,
                    plus_iter.next().expect("bad alignment (-) r"),
                    plus_must_wrap_iter.next().expect("bad wrap info (-) r"),
                    &config.wrap_symbol,
                    line_width.plus,
                    config.wrap_max_lines,
                );

                for i in plus_start..extended_to {
                    new_alignment.push((None, Some(i)));
                }

                (0, extended_to - plus_start)
            }
            (Some(m), Some(p)) => {
                assert_alignment!(m, m_expected, "l (r)");
                assert_alignment!(p, p_expected, "(l) r");

                let (minus_start, m_extended_to) = wrap_text_and_style(
                    &mut new_minus,
                    minus_iter.next().expect("bad alignment l (r)"),
                    minus_must_wrap_iter.next().expect("bad wrap info l (r)"),
                    &config.wrap_symbol,
                    line_width.minus,
                    config.wrap_max_lines,
                );

                let (plus_start, p_extended_to) = wrap_text_and_style(
                    &mut new_plus,
                    plus_iter.next().expect("bad alignment (l) r"),
                    plus_must_wrap_iter.next().expect("bad wrap info (l) r"),
                    &config.wrap_symbol,
                    line_width.plus,
                    config.wrap_max_lines,
                );

                for (new_m, new_p) in (minus_start..m_extended_to).zip(plus_start..p_extended_to) {
                    new_alignment.push((Some(new_m), Some(new_p)));
                }

                // This Some(m):Some(p) alignment might have become uneven, so fill
                // up the shorter side with None.

                let minus_extended = m_extended_to - minus_start;
                let plus_extended = p_extended_to - plus_start;

                let plus_minus = (minus_extended as isize) - (plus_extended as isize);

                if plus_minus < 0 {
                    for n in plus_start + plus_minus.abs() as usize..p_extended_to {
                        new_alignment.push((None, Some(n)));
                    }
                } else if plus_minus > 0 {
                    for n in minus_start + plus_minus as usize..m_extended_to {
                        new_alignment.push((Some(n), None));
                    }
                }

                (minus_extended, plus_extended)
            }
            _ => panic!("unexpected None-None alignment"),
        };

        if minus_extended > 0 {
            new_minus_state.push(State::HunkMinus(None));
            for _ in 1..minus_extended {
                new_minus_state.push(State::HunkMinusWrapped);
            }
        }
        if plus_extended > 0 {
            new_plus_state.push(State::HunkPlus(None));
            for _ in 1..plus_extended {
                new_plus_state.push(State::HunkPlusWrapped);
            }
        }
    }

    (
        new_alignment,
        PlusMinus::new(new_minus_state, new_plus_state),
        PlusMinus::new(new_minus, new_plus),
    )
}

#[allow(clippy::comparison_chain, clippy::type_complexity)]
pub fn wrap_zero_block<'c: 'a, 'a>(
    config: &'c Config,
    mut states: Vec<State>,
    syntax_style_sections: Vec<Vec<(SyntectStyle, &'a str)>>,
    diff_style_sections: Vec<Vec<(Style, &'a str)>>,
    line_numbers_data: &Option<&mut line_numbers::LineNumbersData>,
) -> (
    Vec<State>,
    Vec<Vec<(SyntectStyle, &'a str)>>,
    Vec<Vec<(Style, &'a str)>>,
) {
    // The width is the minimum of the left/right side. The panels should be equally sized,
    // but in rare cases the remaining panel width might differ due to the space the line
    // numbers take up.
    let line_width = if let Some(line_numbers_data) = line_numbers_data {
        let width = available_line_width(&config, &line_numbers_data);
        std::cmp::min(width.minus, width.plus)
    } else {
        std::cmp::min(
            config.side_by_side_data.left_panel.width,
            config.side_by_side_data.right_panel.width,
        )
    };

    // Called with a single line, so no need to use the 1-sized bool vector,
    // if that changes the wrapping logic should to be updated as well.
    assert_eq!(diff_style_sections.len(), 1);

    let (wrap_syntax_style, _) = is_too_long(&syntax_style_sections, line_width);
    let (wrap_diff_style, _) = is_too_long(&diff_style_sections, line_width);

    if wrap_syntax_style || wrap_diff_style {
        let syntax_style = wrap_line(
            syntax_style_sections.into_iter().flatten(),
            &config.wrap_symbol,
            line_width,
            config.wrap_max_lines,
        );
        let diff_style = wrap_line(
            diff_style_sections.into_iter().flatten(),
            &config.wrap_symbol,
            line_width,
            config.wrap_max_lines,
        );

        states.resize_with(syntax_style.len(), || State::HunkZeroWrapped);

        (states, syntax_style, diff_style)
    } else {
        (states, syntax_style_sections, diff_style_sections)
    }
}
#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use syntect::highlighting::Style as SyntectStyle;

    use super::wrap_line;
    use crate::ansi::strip_ansi_codes;
    use crate::style::Style;
    use crate::tests::integration_test_utils::integration_test_utils::{
        make_config_from_args, run_delta,
    };

    lazy_static! {
        static ref S1: Style = Style {
            is_emph: false,
            ..Default::default()
        };
    }
    lazy_static! {
        static ref S2: Style = Style {
            is_emph: true,
            ..Default::default()
        };
    }
    lazy_static! {
        static ref SY: SyntectStyle = SyntectStyle::default();
    }
    lazy_static! {
        static ref SD: Style = Style::default();
    }

    // wrap symbol
    const W: &str = &"+";

    #[test]
    fn test_wrap_line_plain() {
        {
            let line = vec![(*SY, "_0")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*SY, "_0")]]);
        }

        {
            let line = vec![(*S1, "")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "")]]);
        }

        {
            let line = vec![(*S1, "_")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "_")]]);
        }

        {
            let line = vec![(*S1, "_"), (*S2, "0")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "_"), (*S2, "0")]]);
        }

        {
            let line = vec![(*S1, "_012"), (*S2, "34")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "_012"), (*S2, "34")]]);
        }

        {
            let line = vec![(*S1, "_012"), (*S2, "345")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "_012"), (*S2, "345")]]);
        }

        {
            let line = vec![(*S1, "_012"), (*S2, "3456")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(
                lines,
                vec![
                    vec![(*S1, "_012"), (*S2, "34"), (*SD, "+")],
                    vec![(*SD, "_"), (*S2, "56")]
                ]
            );
        }
    }

    #[test]
    fn test_wrap_line_newlines() {
        fn mk_input(len: usize) -> Vec<(Style, &'static str)> {
            const IN: &str = "_0123456789abcdefZ";
            let v = &[*S1, *S2];
            let s1s2 = v.iter().cycle();
            let text: Vec<_> = IN.matches(|_| true).take(len + 1).collect();
            s1s2.zip(text.iter())
                .map(|(style, text)| (style.clone(), *text))
                .collect()
        }
        fn mk_input_nl(len: usize) -> Vec<(Style, &'static str)> {
            const NL: &str = "\n";
            let mut line = mk_input(len);
            line.push((*S2, NL));
            line
        }
        fn mk_expected<'a>(
            prepend: Option<(Style, &'a str)>,
            vec: &Vec<(Style, &'a str)>,
            from: usize,
            to: usize,
            append: Option<(Style, &'a str)>,
        ) -> Vec<(Style, &'a str)> {
            let mut result: Vec<_> = vec[from..to].iter().cloned().collect();
            if let Some(val) = append {
                result.push(val);
            }
            if let Some(val) = prepend {
                result.insert(0, val);
            }
            result
        }

        {
            let line = vec![(*S1, "_012"), (*S2, "345\n")];
            let lines = wrap_line(line, &W, 6, 99);
            assert_eq!(lines, vec![vec![(*S1, "_012"), (*S2, "345\n")]]);
        }

        {
            for i in 0..=6 {
                let line = mk_input(i);
                let lines = wrap_line(line, &W, 6, 99);
                assert_eq!(lines, vec![mk_input(i)]);

                let line = mk_input_nl(i);
                let lines = wrap_line(line, &W, 6, 99);
                assert_eq!(lines, vec![mk_input_nl(i)]);
            }
        }

        {
            let line = mk_input_nl(9);
            let lines = wrap_line(line, &W, 3, 99);
            let expected = mk_input_nl(9);
            let line1 = mk_expected(None, &expected, 0, 3, Some((*SD, &W)));
            let line2 = mk_expected(Some((*SD, "_")), &expected, 3, 5, Some((*SD, &W)));
            let line3 = mk_expected(Some((*SD, "_")), &expected, 5, 7, Some((*SD, &W)));
            let line4 = mk_expected(Some((*SD, "_")), &expected, 7, 11, None);
            assert_eq!(lines, vec![line1, line2, line3, line4]);
        }

        {
            let line = mk_input_nl(10);
            let lines = wrap_line(line, &W, 3, 99);
            let expected = mk_input_nl(10);
            let line1 = mk_expected(None, &expected, 0, 3, Some((*SD, &W)));
            let line2 = mk_expected(Some((*SD, "_")), &expected, 3, 5, Some((*SD, &W)));
            let line3 = mk_expected(Some((*SD, "_")), &expected, 5, 7, Some((*SD, &W)));
            let line4 = mk_expected(Some((*SD, "_")), &expected, 7, 9, Some((*SD, &W)));
            let line5 = mk_expected(Some((*SD, "_")), &expected, 9, 11, Some((*S2, "\n")));
            assert_eq!(lines, vec![line1, line2, line3, line4, line5]);
        }

        {
            let line = vec![(*S1, "_abc"), (*S2, "01230123012301230123"), (*S1, "ZZZZZ")];
            let lines = wrap_line(line.clone(), &W, 4, 1);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines.last().unwrap().last().unwrap().1, "ZZZZZ");
            let lines = wrap_line(line.clone(), &W, 4, 2);
            assert_eq!(lines.len(), 2);
            assert_eq!(lines.last().unwrap().last().unwrap().1, "ZZZZZ");
            let lines = wrap_line(line, &W, 4, 3);
            assert_eq!(lines.len(), 3);
            assert_eq!(lines.last().unwrap().last().unwrap().1, "ZZZZZ");
        }
    }

    #[test]
    fn test_wrap_line_unicode() {
        // from UnicodeSegmentation documentation and the linked
        // // from UnicodeSegmentation documentation
        let line = vec![(*S1, "_abc"), (*S2, "mnö̲"), (*S1, "xyz")];
        let lines = wrap_line(line, &W, 4, 99);
        assert_eq!(
            lines,
            vec![
                vec![(*S1, "_abc"), (*SD, &W)],
                vec![(*SD, "_"), (*S2, "mnö̲"), (*SD, &W)],
                vec![(*SD, "_"), (*S1, "xyz")]
            ]
        );

        // Not working: Tailored grapheme clusters: क्षि  = क् + षि
        let line = vec![(*S1, "_abc"), (*S2, "deநி"), (*S1, "ghij")];
        let lines = wrap_line(line, &W, 4, 99);
        assert_eq!(
            lines,
            vec![
                vec![(*S1, "_abc"), (*SD, &W)],
                vec![(*SD, "_"), (*S2, "deநி"), (*SD, &W)],
                vec![(*SD, "_"), (*S1, "ghij")]
            ]
        );
    }

    const HUNK_ZERO_DIFF: &str = "\
diff --git i/a.py w/a.py
index 223ca50..e69de29 100644
--- i/a.py
+++ w/a.py
@@ -4,3 +15,3 @@
 abcdefghijklmnopqrstuvwxzy 0123456789 0123456789 0123456789 0123456789 0123456789
-a = 1
+a = 2
";

    const HUNK_ZERO_LARGE_LINENUMBERS_DIFF: &str = "\
diff --git i/a.py w/a.py
index 223ca50..e69de29 100644
--- i/a.py
+++ w/a.py
@@ -10,3 +101999,3 @@
 abcdefghijklmnopqrstuvwxzy 0123456789 0123456789 0123456789 0123456789 0123456789
-a = 1
+a = 2
";

    const HUNK_MP_DIFF: &str = "\
diff --git i/a.py w/a.py
index 223ca50..e69de29 100644
--- i/a.py
+++ w/a.py
@@ -4,3 +15,3 @@
 abcdefghijklmnopqrstuvwxzy 0123456789 0123456789 0123456789 0123456789 0123456789
-a = 0123456789 0123456789 0123456789 0123456789 0123456789
+b = 0123456789 0123456789 0123456789 0123456789 0123456789
";

    #[test]
    fn test_wrap_with_linefmt1() {
        // let config = make_config_from_args(&["--side-by-side-wrapped", "--line-numbers-left-format", "│I│", "--line-numbers-right-format", "│WWWW│", "--width", "40"]);
        let mut config = make_config_from_args(&[
            "--side-by-side-wrapped",
            "--line-numbers-left-format",
            "│L│",
            "--line-numbers-right-format",
            "│RRRR│",
            "--width",
            "40",
        ]);
        config.wrap_symbol = "+".into();
        config.truncation_symbol = ">".into();
        let output = run_delta(HUNK_ZERO_DIFF, &config);
        let output = strip_ansi_codes(&output);
        let lines: Vec<_> = output.lines().skip(7).collect();
        let expected = vec![
            "│L│abcdefghijklm+   │RRRR│abcdefghijklm+",
            "│L│nopqrstuvwxzy+   │RRRR│nopqrstuvwxzy+",
            "│L│ 0123456789 0+   │RRRR│ 0123456789 0+",
            "│L│123456789 012+   │RRRR│123456789 012+",
            "│L│3456789 01234567>│RRRR│3456789 01234>",
            "│L│a = 1            │RRRR│a = 2         ",
        ];
        assert_eq!(lines, expected);
    }

    #[test]
    fn test_wrap_with_linefmt2() {
        let mut config = make_config_from_args(&[
            "--side-by-side-wrapped",
            "--line-numbers-left-format",
            "│LLL│",
            "--line-numbers-right-format",
            "│WW {nm} +- {np:2} WW│",
            "--width",
            "60",
        ]);
        config.wrap_symbol = "+".into();
        config.truncation_symbol = ">".into();
        let output = run_delta(HUNK_ZERO_LARGE_LINENUMBERS_DIFF, &config);
        let output = strip_ansi_codes(&output);
        let lines: Vec<_> = output.lines().skip(7).collect();
        let expected = vec![
            "│LLL│abcde+                   │WW   10   +- 101999 WW│abcde+",
            "│LLL│fghij+                   │WW        +-        WW│fghij+",
            "│LLL│klmno+                   │WW        +-        WW│klmno+",
            "│LLL│pqrst+                   │WW        +-        WW│pqrst+",
            "│LLL│uvwxzy 0123456789 012345>│WW        +-        WW│uvwxz>",
            "│LLL│a = 1                    │WW        +- 102000 WW│a = 2 ",
        ];
        assert_eq!(lines, expected);
    }

    #[test]
    fn test_wrap_with_keep_markers() {
        let mut config = make_config_from_args(&[
            "--side-by-side-wrapped",
            "--keep-plus-minus-markers",
            "--width",
            "45",
        ]);
        config.wrap_symbol = "+".into();
        config.truncation_symbol = ">".into();
        let output = run_delta(HUNK_MP_DIFF, &config);
        let output = strip_ansi_codes(&output);
        let lines: Vec<_> = output.lines().skip(7).collect();
        let expected = vec![
            "│ 4  │ abcdefghijklmn+│ 15 │ abcdefghijklmn+",
            "│    │ opqrstuvwxzy 0+│    │ opqrstuvwxzy 0+",
            "│    │ 123456789 0123+│    │ 123456789 0123+",
            "│    │ 456789 0123456+│    │ 456789 0123456+",
            "│    │ 789 0123456789>│    │ 789 0123456789>",
            "│ 5  │-a = 0123456789+│ 16 │+b = 0123456789+",
            "│    │  0123456789 01+│    │  0123456789 01+",
            "│    │ 23456789 01234+│    │ 23456789 01234+",
            "│    │ 56789 01234567+│    │ 56789 01234567+",
            "│    │ 89             │    │ 89             ",
        ];
        assert_eq!(lines, expected);
    }
}
