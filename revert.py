import json

data = json.load(open("radiology_workflow.json"))

if "workflowPieces" in data:
    # Oops it's still missing the outer envelope.
    print("It is the flat one.")
