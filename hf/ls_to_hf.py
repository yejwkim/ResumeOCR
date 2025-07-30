import json

IN = "data/annotations/annotations.json"
OUT = "data/hf_dataset.jsonl"

def close(a, b, tol = 1e-4):
    return abs(a - b) <= tol

def same_box(b1, b2, tol = 1e-4):
    return all(close(x, y, tol) for x, y in zip(b1, b2))

with open(IN) as f_in, open(OUT, "w") as f_out:
    data = json.load(f_in)
    print(f"# of images: {len(data)}")
    issue_found = False

    for entry in data: # Single page
        raw_url = entry["data"]['image']
        image_path = raw_url.replace("http://localhost:8080", "data/images")
        ocr_data = entry["data"]['ocr_data']
        annotations = entry["annotations"][0]["result"]
        id = entry["annotations"][0]["id"]
        pairs, unmatched = [], []
        
        for ann in annotations: # Single annotation
            v = ann["value"]
            x1, y1, width, height = v["x"], v["y"], v["width"], v["height"]
            x2, y2 = x1 + width, y1 + height
            ann_box = [x1, y1, x2, y2]
            
            match = next((o for o in ocr_data if same_box(o["bbox"], ann_box)), None)
            if match:
                pairs.append((match, ann))
            else:
                unmatched.append(ann_box)
        
        if unmatched:
            print(f"[!] {len(unmatched)} boxes in {image_path} with ID {id} failed to match OCR tokens.")
            continue

        words, bboxes, labels = [], [], []
        for ocr, ann in pairs:
            word = ocr.get("text", "")
            if not word:
                print("AAA")
                continue
            words.append(word)
            bboxes.append(ocr["bbox"])
            label = ann["value"].get("rectanglelabels") or []
            labels.append(label[0])
            # labels.append(label[0] if label else "O")
        if words:
            hf = {"image_path": image_path, "words": words, "bboxes": bboxes, "labels": labels}
            f_out.write(json.dumps(hf) + "\n")
