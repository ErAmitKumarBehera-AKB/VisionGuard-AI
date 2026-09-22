import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_casting_image(is_defective, size=(224, 224)):
    # Create background (dark metallic gray)
    bg_color = random.randint(30, 50)
    img = Image.new("RGB", size, (bg_color, bg_color, bg_color))
    draw = ImageDraw.Draw(img)
    
    # Draw circular casting piece (metallic disc)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 2 - 20
    
    # Metal disc color
    metal_color = random.randint(160, 200)
    draw.ellipse(
        [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
        fill=(metal_color, metal_color, metal_color),
        outline=(metal_color - 30, metal_color - 30, metal_color - 30),
        width=3
    )
    
    # Draw inner features (spokes, inner ring to make it look like an impeller)
    inner_radius = radius // 2
    draw.ellipse(
        [center[0] - inner_radius, center[1] - inner_radius, center[0] + inner_radius, center[1] + inner_radius],
        outline=(metal_color - 40, metal_color - 40, metal_color - 40),
        width=2
    )
    
    # Draw 4 spokes
    for angle in [0, 45, 90, 135]:
        rad = np.radians(angle)
        x1 = center[0] - int(radius * np.cos(rad))
        y1 = center[1] - int(radius * np.sin(rad))
        x2 = center[0] + int(radius * np.cos(rad))
        y2 = center[1] + int(radius * np.sin(rad))
        draw.line((x1, y1, x2, y2), fill=(metal_color - 30, metal_color - 30, metal_color - 30), width=4)
        
    # Draw center hole
    hole_radius = 10
    draw.ellipse(
        [center[0] - hole_radius, center[1] - hole_radius, center[0] + hole_radius, center[1] + hole_radius],
        fill=(bg_color, bg_color, bg_color)
    )
    
    # Add random metal texture noise
    img_arr = np.array(img)
    noise = np.random.normal(0, 10, img_arr.shape).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    
    if is_defective:
        # Draw a defect
        defect_type = random.choice(["crack", "void", "scratch"])
        if defect_type == "crack":
            # Draw a jagged dark line
            start_x = random.randint(center[0] - radius + 15, center[0] + radius - 15)
            start_y = random.randint(center[1] - radius + 15, center[1] + radius - 15)
            curr_x, curr_y = start_x, start_y
            for _ in range(5):
                next_x = curr_x + random.randint(-15, 15)
                next_y = curr_y + random.randint(-15, 15)
                draw.line((curr_x, curr_y, next_x, next_y), fill=(10, 10, 10), width=random.randint(2, 4))
                curr_x, curr_y = next_x, next_y
        elif defect_type == "void":
            # Draw a dark spot (casting blowhole)
            vx = random.randint(center[0] - radius + 20, center[0] + radius - 20)
            vy = random.randint(center[1] - radius + 20, center[1] + radius - 20)
            vr = random.randint(5, 12)
            draw.ellipse([vx - vr, vy - vr, vx + vr, vy + vr], fill=(20, 20, 20))
        elif defect_type == "scratch":
            # Draw a thin light scratch
            sx1 = random.randint(center[0] - radius + 20, center[0] + radius - 20)
            sy1 = random.randint(center[1] - radius + 20, center[1] + radius - 20)
            sx2 = sx1 + random.randint(-40, 40)
            sy2 = sy1 + random.randint(-40, 40)
            draw.line((sx1, sy1, sx2, sy2), fill=(240, 240, 240), width=1)
            
    # Apply a slight blur to make it look realistic
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img

def generate_dataset(base_dir="data/raw"):
    categories = ["ok", "defective"]
    splits = {"train": 100, "test": 20}
    
    for split, count in splits.items():
        for category in categories:
            folder = os.path.join(base_dir, split, category)
            os.makedirs(folder, exist_ok=True)
            
            is_defective = (category == "defective")
            print(f"Generating {count} images for {split}/{category}...")
            for i in range(count):
                img = create_casting_image(is_defective)
                img.save(os.path.join(folder, f"casting_{i:04d}.png"))
                
    print("Dataset generation complete!")

if __name__ == "__main__":
    generate_dataset()
