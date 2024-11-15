from docling.document_converter import DocumentConverter
import pdfplumber
import base64
import os

# Initialize DocumentConverter for docling
converter = DocumentConverter()
source_path = "skincancer.pdf"

# Parse text and tables with docling
docling_result = converter.convert(source_path)
markdown_content = docling_result.document.export_to_markdown()

# Save the Markdown content (text + tables) to a file
with open("parsed_text_and_tables.md", "w", encoding="utf-8") as text_file:
    text_file.write(markdown_content)

# Parse images with pdfplumber
image_dir = "extracted_images"
os.makedirs(image_dir, exist_ok=True)

# images with bounding boxes within the page dimensions are being successfully cropped and saved to extracted_images
# Images with bounding boxes that fall outside the page boundaries are being skipped, which prevents the script from encountering the ValueError.
with pdfplumber.open(source_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        for image_idx, img in enumerate(page.images):
            # Extract the image's location and content
            img_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
            
            # Check if the bounding box is within the page bounds
            page_width, page_height = page.width, page.height
            if (
                0 <= img_bbox[0] <= page_width and
                0 <= img_bbox[1] <= page_height and
                0 <= img_bbox[2] <= page_width and
                0 <= img_bbox[3] <= page_height
            ):
                # Crop and save the image
                cropped_image = page.within_bbox(img_bbox).to_image()
                image_filename = f"{image_dir}/image_{page_num}_{image_idx + 1}.png"
                cropped_image.save(image_filename)
                print(f"Image saved: {image_filename}")
            else:
                print(f"Skipped image on page {page_num} with out-of-bounds bbox: {img_bbox}")
