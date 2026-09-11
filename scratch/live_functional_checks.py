import sys
import urllib.request
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

def post_chat(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("Checking backend health...")
with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
    print("Health response:", resp.read().decode("utf-8"))

print("\n--- TEST A: SINGLE PRODUCT ---")
turn1 = post_chat({"message": "Analyze barcode 3017620422003"})
print("Turn 1 Barcode Returned:", turn1.get("barcode"))
print("Turn 1 Response Snippet:", turn1.get("response")[:120].encode('ascii', errors='ignore').decode('ascii'))

active_barcode = turn1.get("barcode")
turn2 = post_chat({
    "message": "What about the sugar?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": turn1["response"]}
    ],
    "barcode": active_barcode
})
print("Turn 2 Barcode Returned:", turn2.get("barcode"))
print("Turn 2 Response Snippet:", turn2.get("response")[:120].encode('ascii', errors='ignore').decode('ascii'))

turn3 = post_chat({
    "message": "What about protein?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": turn1["response"]},
        {"role": "user", "content": "What about the sugar?"},
        {"role": "assistant", "content": turn2["response"]}
    ],
    "barcode": active_barcode
})
print("Turn 3 Barcode Returned:", turn3.get("barcode"))
print("Turn 3 Response Snippet:", turn3.get("response")[:120].encode('ascii', errors='ignore').decode('ascii'))

print("\n--- TEST B: COMPARISON ---")
comp1 = post_chat({"message": "Compare 3017620422003 and 0051500255162"})
print("Comp 1 Barcodes Returned:", comp1.get("comparison_barcodes"))

comp2 = post_chat({
    "message": "Which one has less sugar?",
    "messages": [
        {"role": "user", "content": "Compare 3017620422003 and 0051500255162"},
        {"role": "assistant", "content": comp1["response"]}
    ],
    "barcodes": comp1.get("comparison_barcodes")
})
print("Comp 2 Barcodes Returned:", comp2.get("comparison_barcodes"))
print("Comp 2 Response Snippet:", comp2.get("response")[:120].encode('ascii', errors='ignore').decode('ascii'))

print("\n--- TEST F: GENERAL RAG AFTER PRODUCT ---")
rag_turn = post_chat({
    "message": "Why is sesame considered a major food allergen?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": turn1["response"]}
    ],
    "barcode": active_barcode
})
print("RAG Response Snippet:", rag_turn.get("response")[:120].encode('ascii', errors='ignore').decode('ascii'))

print("\nALL LIVE FUNCTIONAL CHECKS COMPLETED SUCCESSFULLY!")
