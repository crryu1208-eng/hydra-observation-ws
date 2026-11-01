# Claude Code memory (snapshot)

Copy of the Claude Code project memory for this workspace
(`~/.claude/projects/-home-etri-hydra-ws/memory/`). Claude reads the live copy there, not this one;
this snapshot is kept for history and for restoring the memory on another machine:

```bash
mkdir -p ~/.claude/projects/-home-etri-hydra-ws/memory
cp docs/claude_memory/*.md ~/.claude/projects/-home-etri-hydra-ws/memory/
rm ~/.claude/projects/-home-etri-hydra-ws/memory/README.md
```

The project folder name is derived from the workspace path (`/home/etri/hydra_ws`); adjust it if the
workspace lives elsewhere.
