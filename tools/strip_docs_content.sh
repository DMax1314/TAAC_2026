#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCS_ROOT="$REPO_ROOT/docs"

MODE="write"
PRINT_ONLY="0"

usage() {
    cat <<'EOF'
Usage:
  bash tools/strip_docs_content.sh [--dry-run] [--print] <target> [<target> ...]

Targets:
  - A Markdown file under docs/
  - A directory under docs/; all *.md files inside it will be processed recursively

Accepted target forms:
  - docs/guide/local-site.md
  - guide/local-site.md
  - guide
  - /absolute/path/to/repo/docs/guide

Behavior:
  - Keeps only the top YAML frontmatter block (if present) and Markdown ATX headings (#, ##, ### ...)
  - Removes paragraphs, lists, tables, blockquotes, code blocks, images, and other body content
  - Ignores # lines inside fenced code blocks

Options:
  --dry-run  Preview matched files without writing changes
  --print    Print the transformed content for exactly one Markdown file and exit
  --help     Show this help message

Examples:
  bash tools/strip_docs_content.sh docs/guide/local-site.md
  bash tools/strip_docs_content.sh guide
  bash tools/strip_docs_content.sh --dry-run docs/papers
  bash tools/strip_docs_content.sh --print docs/architecture.md
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

resolve_target_path() {
    local raw_path="$1"
    local candidate=""

    if [[ "$raw_path" == /* ]]; then
        candidate="$raw_path"
    elif [[ -e "$REPO_ROOT/$raw_path" ]]; then
        candidate="$REPO_ROOT/$raw_path"
    elif [[ -e "$DOCS_ROOT/$raw_path" ]]; then
        candidate="$DOCS_ROOT/$raw_path"
    else
        die "target not found: $raw_path"
    fi

    if [[ -d "$candidate" ]]; then
        candidate="$(cd -- "$candidate" && pwd)"
    else
        candidate="$(cd -- "$(dirname -- "$candidate")" && pwd)/$(basename -- "$candidate")"
    fi

    case "$candidate" in
        "$DOCS_ROOT"|"$DOCS_ROOT"/*)
            ;;
        *)
            die "target must be inside docs/: $raw_path"
            ;;
    esac

    printf '%s\n' "$candidate"
}

collect_markdown_files() {
    local target_path="$1"

    if [[ -f "$target_path" ]]; then
        [[ "$target_path" == *.md ]] || die "file is not a Markdown document: $target_path"
        printf '%s\n' "$target_path"
        return
    fi

    if [[ -d "$target_path" ]]; then
        find "$target_path" -type f -name '*.md' | LC_ALL=C sort
        return
    fi

    die "unsupported target: $target_path"
}

strip_markdown_to_structure() {
    local source_path="$1"
    local output_path="$2"

    awk '
    function emit_pending_blank() {
        if (pending_blank && emitted_any) {
            print ""
            pending_blank = 0
        }
    }

    NR == 1 && $0 == "---" {
        in_frontmatter = 1
        emitted_any = 1
        print $0
        next
    }

    in_frontmatter {
        print $0
        if ($0 == "---") {
            in_frontmatter = 0
            pending_blank = 1
        }
        next
    }

    {
        line = $0

        if (line ~ /^(```|~~~)/) {
            in_code_fence = !in_code_fence
            next
        }

        if (!in_code_fence && line ~ /^#+([[:space:]]+|$)/) {
            emit_pending_blank()
            print line
            emitted_any = 1
            pending_blank = 1
            next
        }

        if (line ~ /^[[:space:]]*$/) {
            pending_blank = 1
        }
    }
    ' "$source_path" > "$output_path"
}

relative_repo_path() {
    local path="$1"
    printf '%s\n' "${path#$REPO_ROOT/}"
}

print_summary() {
    local mode_label="$1"
    local matched_files="$2"
    local changed_files="$3"
    local unchanged_files="$4"
    local failures="$5"

    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'docs_root=%s\n' "$DOCS_ROOT"
    printf 'mode=%s\n' "$mode_label"
    printf 'matched_files=%s\n' "$matched_files"
    printf 'changed_files=%s\n' "$changed_files"
    printf 'unchanged_files=%s\n' "$unchanged_files"
    printf 'failures=%s\n' "$failures"
}

main() {
    local -a input_targets=()

    while (($# > 0)); do
        case "$1" in
            --dry-run)
                MODE="dry-run"
                ;;
            --print)
                PRINT_ONLY="1"
                MODE="dry-run"
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --*)
                die "unknown option: $1"
                ;;
            *)
                input_targets+=("$1")
                ;;
        esac
        shift
    done

    ((${#input_targets[@]} > 0)) || {
        usage
        exit 2
    }

    local -A seen_files=()
    local -a markdown_files=()
    local target=""
    local resolved_target=""
    local matched_file=""

    for target in "${input_targets[@]}"; do
        resolved_target="$(resolve_target_path "$target")"
        while IFS= read -r matched_file; do
            [[ -n "$matched_file" ]] || continue
            if [[ -z "${seen_files[$matched_file]+x}" ]]; then
                seen_files[$matched_file]="1"
                markdown_files+=("$matched_file")
            fi
        done < <(collect_markdown_files "$resolved_target")
    done

    ((${#markdown_files[@]} > 0)) || die "no Markdown files matched"

    if ((PRINT_ONLY)) && ((${#markdown_files[@]} != 1)); then
        die "--print requires exactly one Markdown file target"
    fi

    local matched_files="${#markdown_files[@]}"
    local changed_files="0"
    local unchanged_files="0"
    local failures="0"
    local source_path=""
    local tmp_path=""
    local relative_path=""
    local headings_count="0"
    local frontmatter_present="false"

    for source_path in "${markdown_files[@]}"; do
        tmp_path="$(mktemp)"
        if ! strip_markdown_to_structure "$source_path" "$tmp_path"; then
            rm -f "$tmp_path"
            failures="$((failures + 1))"
            printf '[failed] %s\n' "$(relative_repo_path "$source_path")" >&2
            continue
        fi

        if ((PRINT_ONLY)); then
            cat "$tmp_path"
            rm -f "$tmp_path"
            exit 0
        fi

        relative_path="$(relative_repo_path "$source_path")"
        headings_count="$(grep -Ec '^#+([[:space:]]+|$)' "$tmp_path" || true)"
        if [[ -s "$tmp_path" ]] && [[ "$(head -n 1 "$tmp_path")" == '---' ]]; then
            frontmatter_present="true"
        else
            frontmatter_present="false"
        fi

        if cmp -s "$source_path" "$tmp_path"; then
            unchanged_files="$((unchanged_files + 1))"
            printf '[skip] %s headings=%s frontmatter=%s\n' "$relative_path" "$headings_count" "$frontmatter_present"
            rm -f "$tmp_path"
            continue
        fi

        changed_files="$((changed_files + 1))"
        if [[ "$MODE" == 'dry-run' ]]; then
            printf '[dry-run] %s headings=%s frontmatter=%s\n' "$relative_path" "$headings_count" "$frontmatter_present"
            rm -f "$tmp_path"
            continue
        fi

        mv "$tmp_path" "$source_path"
        printf '[updated] %s headings=%s frontmatter=%s\n' "$relative_path" "$headings_count" "$frontmatter_present"
    done

    print_summary "$MODE" "$matched_files" "$changed_files" "$unchanged_files" "$failures"

    if ((failures > 0)); then
        exit 1
    fi
}

main "$@"