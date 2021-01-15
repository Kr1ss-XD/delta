#!/bin/bash

__delta_previous_extglob_setting=$(shopt -p extglob)
shopt -s extglob

__delta_complete_commands() {
    COMPREPLY=( $(compgen -W "${commands[*]}" -- "$cur") )
}

_delta_delta() {
    __delta_complete_commands
}

_delta() {
    local previous_extglob_setting=$(shopt -p extglob)
    shopt -s extglob

    local commands=(
        --24-bit-color
        --color-only
        --commit-decoration-style
        --commit-style
        --dark
        --diff-highlight
        --diff-so-fancy
        --features
        --file-added-label
        --file-copied-label
        --file-decoration-style
        --file-modified-label
        --file-removed-label
        --file-renamed-label
        --file-style
        --help -h
        --hunk-header-decoration-style
        --hunk-header-file-style
        --hunk-header-line-number-style
        --hunk-header-style
        --hyperlinks
        --hyperlinks-file-link-format
        --inspect-raw-lines
        --keep-plus-minus-markers
        --light
        --line-buffer-size
        --line-numbers -n
        --line-numbers-left-format
        --line-numbers-left-style
        --line-numbers-minus-style
        --line-numbers-plus-style
        --line-numbers-right-format
        --line-numbers-right-style
        --line-numbers-zero-style
        --list-languages
        --list-syntax-themes
        --max-line-distance
        --max-line-length
        --minus-emph-style
        --minus-empty-line-marker-style
        --minus-non-emph-style
        --minus-style
        --navigate
        --no-gitconfig
        --paging
        --plus-emph-style
        --plus-empty-line-marker-style
        --plus-non-emph-style
        --plus-style
        --raw
        --show-config
        --show-syntax-themes
        --side-by-side -s
        --syntax-theme
        --tabs
        --version -V
        --whitespace-error-style
        --width -w
        --word-diff-regex
        --zero-style
    )

    COMPREPLY=()
    local cur prev words cword
    _get_comp_words_by_ref -n : cur prev words cword

    local command='delta' command_pos=0
    local counter=1
    while [ $counter -lt $cword ]; do
      case "${words[$counter]}" in
          *)
              command="${words[$counter]}"
              command_pos=$counter
              break
              ;;
      esac
      (( counter++ ))
    done

    local completions_func=_delta_${command}

    declare -F $completions_func >/dev/null && $completions_func

    eval "$previous_extglob_setting"
    return 0
}

eval "$__delta_previous_extglob_setting"
unset __delta_previous_extglob_setting

complete -F _delta delta
