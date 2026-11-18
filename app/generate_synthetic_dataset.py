"""Build the project's 100-row synthetic training corpus deterministically."""
import json
from pathlib import Path


CATALOG = {
    "Electronics": [
        ("Wireless Headphones", "Sound", ["Bluetooth 5.3", "Bluetooth 5.2", "Bluetooth 5.0", "Bluetooth 5.1"], "Battery", ["Up to 40 hours", "Up to 30 hours", "Up to 24 hours", "Up to 18 hours"], "Design", ["Foldable over-ear", "Lightweight on-ear", "Padded over-ear", "Compact foldable"]),
        ("Desk Lamp", "Brightness", ["Three brightness levels", "Dimmable LED", "Four brightness settings", "Two brightness levels"], "Light", ["Warm and cool modes", "Warm white", "Adjustable colour temperature", "Neutral white"], "Power", ["USB-C input", "USB powered", "12 W adapter", "USB-A input"]),
        ("Portable Speaker", "Connection", ["Bluetooth 5.2", "Bluetooth 5.3", "Bluetooth 5.0", "Bluetooth 5.1"], "Playback", ["Up to 12 hours", "Up to 10 hours", "Up to 8 hours", "Up to 15 hours"], "Body", ["Compact design", "Fabric finish", "Carry loop", "Small travel-ready body"]),
        ("Mechanical Keyboard", "Layout", ["87-key tenkeyless", "104-key full size", "68-key compact", "75% layout"], "Switches", ["Tactile mechanical", "Linear mechanical", "Quiet tactile", "Mechanical switches"], "Connection", ["USB-C wired", "Detachable USB-C", "Wired USB-A", "USB-C wired"]),
    ],
    "Fashion": [
        ("Everyday T-Shirt", "Fabric", ["100% cotton", "Cotton jersey", "Cotton blend", "Soft knit cotton"], "Fit", ["Regular fit", "Relaxed fit", "Classic fit", "Straight fit"], "Care", ["Machine wash cold", "Machine wash", "Wash with similar colours", "Gentle machine wash"]),
        ("Canvas Backpack", "Capacity", ["20 litres", "18 litres", "24 litres", "16 litres"], "Material", ["Cotton canvas", "Recycled polyester", "Polyester canvas", "Woven canvas"], "Storage", ["Padded laptop sleeve", "Two main compartments", "Front zip pocket", "Internal organiser pocket"]),
        ("Running Socks", "Material", ["Cotton-polyester blend", "Nylon blend", "Moisture-wicking polyester blend", "Cotton and elastane"], "Pack", ["Three pairs", "Two pairs", "Five pairs", "Four pairs"], "Length", ["Ankle length", "Crew length", "Quarter length", "No-show length"]),
        ("Everyday Cap", "Material", ["Cotton twill", "Polyester blend", "Brushed cotton", "Lightweight cotton"], "Closure", ["Adjustable strap", "Metal buckle", "Hook-and-loop strap", "Snapback closure"], "Brim", ["Curved brim", "Pre-curved brim", "Flat brim", "Structured curved brim"]),
    ],
    "Home & Kitchen": [
        ("French Press", "Capacity", ["1 litre", "800 ml", "600 ml", "1.2 litres"], "Material", ["Borosilicate glass and stainless steel", "Glass and stainless steel", "Borosilicate glass", "Stainless steel body"]),
        ("Glass Storage Set", "Pieces", ["10 containers with lids", "Set of 8 containers", "12-piece set", "6 containers with lids"], "Material", ["Borosilicate glass", "Tempered glass", "Clear glass", "Borosilicate glass"], "Safe use", ["Freezer safe; microwave without lids", "Microwave safe without lids", "Freezer safe", "Microwave safe without lids"]),
        ("Cotton Towel Set", "Pieces", ["Set of 4", "Set of 2", "Set of 6", "Set of 3"], "Material", ["100% cotton", "Cotton terry", "Ring-spun cotton", "Cotton blend"], "Care", ["Machine washable", "Wash before first use", "Machine wash warm", "Tumble dry low"]),
        ("Ceramic Dinner Plate", "Diameter", ["26 cm", "28 cm", "23 cm", "20 cm"], "Material", ["Glazed ceramic", "Stoneware", "Porcelain", "Ceramic"], "Care", ["Dishwasher safe", "Hand wash recommended", "Microwave safe", "Dishwasher safe"]),
    ],
    "Fitness": [
        ("Yoga Mat", "Dimensions", ["183 x 61 cm", "173 x 61 cm", "183 x 66 cm", "180 x 60 cm"], "Thickness", ["6 mm", "5 mm", "8 mm", "4 mm"], "Material", ["Textured TPE", "Natural rubber", "PVC foam", "NBR foam"]),
        ("Resistance Band Set", "Bands", ["Five resistance levels", "Set of 3 bands", "Four resistance levels", "Set of 5 bands"], "Material", ["Latex", "Natural latex", "Fabric blend", "Thermoplastic elastomer"], "Included", ["Carry pouch", "Storage bag", "Exercise guide", "Mesh pouch"]),
        ("Steel Water Bottle", "Capacity", ["750 ml", "600 ml", "1 litre", "500 ml"], "Material", ["Stainless steel", "Double-wall stainless steel", "Food-grade stainless steel", "18/8 stainless steel"], "Lid", ["Screw-top lid", "Flip-top lid", "Leak-resistant screw lid", "Wide-mouth cap"]),
        ("Foam Roller", "Length", ["45 cm", "30 cm", "60 cm", "33 cm"], "Surface", ["Textured foam", "Smooth foam", "Ridged surface", "Dimpled surface"], "Material", ["EVA foam", "High-density foam", "Polypropylene core with foam", "EPP foam"]),
    ],
    "Stationery": [
        ("Hardcover Notebook", "Pages", ["192 ruled pages", "160 dotted pages", "200 lined pages", "120 blank pages"], "Paper", ["90 gsm paper", "80 gsm paper", "Cream paper", "Recycled paper"], "Binding", ["Thread-bound", "Case-bound", "Lay-flat binding", "Hardcover binding"]),
        ("Gel Pen Set", "Quantity", ["Pack of 12", "Pack of 6", "Set of 10", "Pack of 8"], "Tip", ["0.5 mm tip", "0.7 mm tip", "Fine 0.38 mm tip", "Medium 1.0 mm tip"], "Ink", ["Black gel ink", "Blue gel ink", "Assorted ink colours", "Black and blue ink"]),
        ("Desk Organiser", "Compartments", ["Five compartments", "Three sections", "Six compartments", "Four compartments"], "Material", ["Bamboo", "Acrylic", "Powder-coated steel", "Recycled plastic"], "Size", ["Compact desktop size", "28 x 18 cm", "Two-tier design", "Fits standard stationery"]),
        ("Watercolour Pad", "Sheets", ["24 sheets", "30 sheets", "12 sheets", "20 sheets"], "Paper", ["300 gsm cold-press paper", "200 gsm textured paper", "Acid-free watercolour paper", "Cold-press paper"], "Binding", ["Spiral bound", "Gummed edge", "Top-bound pad", "Hard-backed pad"]),
    ],
}


def build_records():
    records = []
    for category, products in CATALOG.items():
        for group, product in enumerate(products):
            name, key1, vals1, key2, vals2 = product[:5]
            key3, vals3 = product[5:7] if len(product) > 5 else ("Finish", ["Neutral colour", "Simple finish", "Everyday design", "Compact form"])
            for i in range(5):
                specs = {key1: vals1[(i + group) % len(vals1)], key2: vals2[(i * 2 + group) % len(vals2)], key3: vals3[(i + 2 * group) % len(vals3)]}
                title = f"{['Everyday', 'Classic', 'Compact', 'Studio', 'Essential'][i]} {name} {group * 5 + i + 1:02d}"
                values = list(specs.values())
                openings = [f"Bring {title} into your routine with a practical combination of {values[0].lower()} and {values[1].lower()}.",
                            f"Designed for everyday use, {title} pairs {values[0].lower()} with {values[1].lower()}.",
                            f"Keep your setup simple with {title}, featuring {values[0].lower()} alongside {values[1].lower()}.",
                            f"{title} brings together {values[0].lower()} and {values[1].lower()} in a straightforward {category.lower()} essential.",
                            f"For daily tasks and routines, {title} offers {values[0].lower()} and {values[1].lower()}."]
                description = openings[i] + f" {key3.replace('_', ' ').lower().capitalize()}: {values[2]}. The listed specifications make it easy to see whether it suits your needs."
                records.append({"id": f"syn-{len(records)+1:03d}", "category": category, "product_name": title,
                                "specs": specs, "description": description})
    return records


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "products.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build_records()), encoding="utf-8")
    print(f"Wrote {len(build_records())} synthetic examples to {output}")
