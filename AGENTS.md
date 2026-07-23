# AGENTS.md

This file is auto-loaded by coding agents at session start.
Project: D:\Nature_based_SE\ (Agent Web Search — self-hosted search API for AI agents).

## Progress tracking

Plan, decisions, and progress live in an Obsidian vault, linked here as a junction:

  @.vault/  →  D:\Obsidian\AI-Search-API

`.vault` is a directory junction. Edits to `.vault/*` write through to the real vault files.
DO NOT delete the junction with `Remove-Item -Recurse` — that only removes the link, not the
vault, but use `Remove-Item` (no `-Recurse`) to be safe, OR `cmd /c rmdir .vault`.

## Session startup (mandatory)

Read these in order before doing any work:

1. `.vault/README.md`              — bird's-eye view of the project
2. `.vault/Resume-Protocol.md`    — exact "where to resume" pointer
3. `.vault/Decisions-Log.md`      — all confirmed architectural decisions

If the user pastes a "Project Brief" at session start, read that too — it overrides.

## During the session

- Short answers; detail only when asked.
- Code goes in `D:\Nature_based_SE\`. Notes/progress go in `.vault/`.

## After each completed stage / confirmed decision

Update the vault yourself (don't wait to be asked):

1. **Decisions-Log.md** — append new decision at the TOP (newest first). Format: `## D-NNN — Title` with the rationale. Move resolved items out of "Pending decisions" at the bottom.
2. **01-Pipeline-Map.md** + **README.md** — toggle the stage's status marker:
   - `⬜` not started → `⏳` in progress → `✅` done
3. **Resume-Protocol.md** — move the "Next topic" pointer down to the next pending item.
4. **04-Fetch-Pending-Topics.md** (or the relevant stage file) — mark the just-taught sub-topic ✅ and add a one-line "What was decided" note.
5. **Interview-Notes.md** — if the decision is a senior/junior distinction, append a bullet.

## Pending-state hygiene

At the end of any session that ends mid-stage, append a one-liner to the bottom of
`Resume-Protocol.md` under a `## Session log` heading:
`- YYYY-MM-DD: stopped after <topic>; next: <pointer>`

## Pitfalls (do not repeat)

- `local://` paths are session-scoped — they die with the session. Anything meant to persist
  must live in `D:\Nature_based_SE\` or `.vault/`.
- This project is NOT a git repo. Do not run git commands.
- Git Bash `VAR=val command` does NOT pass env vars to native Windows executables (powershell.exe, cmd.exe, node.exe, etc.). Use `export VAR=val && command` instead.
