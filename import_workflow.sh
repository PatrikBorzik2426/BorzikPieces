#!/bin/bash
# Import radiology_workflow.json into the running Domino instance.
# Safe to re-run — each run creates a new workflow entry (delete old ones in the UI if needed).
set -e

API="http://localhost:8000"
WORKFLOW_FILE="${1:-radiology_workflow.json}"

if [[ ! -f "$WORKFLOW_FILE" ]]; then
  echo "ERROR: $WORKFLOW_FILE not found. Run from the BorzikPieces project root."
  exit 1
fi

echo "=== Authenticating ==="
TOKEN=$(curl -s -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@email.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: Failed to get token. Is the stack running? (docker compose up -d)"
  exit 1
fi
echo "Token obtained."

echo "=== Importing $WORKFLOW_FILE ==="
RESPONSE=$(curl -s -X POST "$API/workspaces/1/workflows" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @"$WORKFLOW_FILE")

ID=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('id','ERROR'))" 2>/dev/null)
NAME=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('name','?'))" 2>/dev/null)

if [[ "$ID" == "ERROR" || -z "$ID" ]]; then
  echo "ERROR: Import failed."
  echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
  exit 1
fi

echo ""
echo "Workflow imported successfully."
echo "  ID:   $ID"
echo "  Name: $NAME"
echo ""
echo "Open http://localhost:3000 and log in with admin@email.com / admin"
