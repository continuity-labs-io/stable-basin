#!/bin/bash

mkdir -p .git/hooks

cat << 'EOF' > .git/hooks/pre-commit
#!/bin/bash

echo "Running Multi-Agent Auditor..."
python tools/multi_agent_auditor.py
if [ $? -ne 0 ]; then
    echo "Commit rejected by Multi-Agent Auditor."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
echo "Git pre-commit hook installed successfully."
