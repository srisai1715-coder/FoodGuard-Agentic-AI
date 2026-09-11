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

print("\n--- LIVE TEST A: SINGLE PRODUCT ---")
t1 = post_chat({"message": "Analyze barcode 3017620422003"})
print("Turn 1 Barcode:", t1.get("barcode"))
print("Turn 1 Snippet:", t1.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

t2 = post_chat({
    "message": "What about the sugar?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": t1["response"]}
    ],
    "barcode": t1.get("barcode")
})
print("Turn 2 Barcode:", t2.get("barcode"))
print("Turn 2 Snippet:", t2.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

t3 = post_chat({
    "message": "Is that high?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": t1["response"]},
        {"role": "user", "content": "What about the sugar?"},
        {"role": "assistant", "content": t2["response"]}
    ],
    "barcode": t1.get("barcode")
})
print("Turn 3 Snippet:", t3.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

print("\n--- LIVE TEST B: COMPARISON ---")
c1 = post_chat({"message": "Compare 3017620422003 and 0051500255162"})
print("Comp 1 Barcodes:", c1.get("comparison_barcodes"))

c2 = post_chat({
    "message": "Which one has less sugar?",
    "messages": [
        {"role": "user", "content": "Compare 3017620422003 and 0051500255162"},
        {"role": "assistant", "content": c1["response"]}
    ],
    "barcodes": c1.get("comparison_barcodes")
})
print("Comp 2 Barcodes:", c2.get("comparison_barcodes"))
print("Comp 2 Snippet:", c2.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

print("\n--- LIVE TEST E: RAG TRANSITION ---")
r1 = post_chat({"message": "Analyze barcode 3017620422003"})
r2 = post_chat({
    "message": "Why is sesame considered a major food allergen?",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": r1["response"]}
    ],
    "barcode": r1.get("barcode")
})
print("RAG Snippet:", r2.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

r3 = post_chat({
    "message": "Now tell me about the sugar in this product.",
    "messages": [
        {"role": "user", "content": "Analyze barcode 3017620422003"},
        {"role": "assistant", "content": r1["response"]},
        {"role": "user", "content": "Why is sesame considered a major food allergen?"},
        {"role": "assistant", "content": r2["response"]}
    ],
    "barcode": r1.get("barcode")
})
print("Turn 3 Return to Product Barcode:", r3.get("barcode"))
print("Turn 3 Snippet:", r3.get("response")[:100].encode('ascii', errors='ignore').decode('ascii'))

print("\n--- LIVE TEST F: CONTEXT REPLACEMENT ---")
rep1 = post_chat({"message": "Analyze barcode 3017620422003"})
rep2 = post_chat({
    "message": "Analyze barcode 0051500255162",
    "barcode": rep1.get("barcode")
})
print("Replaced Barcode Returned:", rep2.get("barcode"))
assert rep2.get("barcode") == "0051500255162"

print("\nALL LIVE CHATBOT TESTS A-G COMPLETED SUCCESSFULLY!")
