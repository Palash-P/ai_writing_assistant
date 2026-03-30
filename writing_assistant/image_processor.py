# writing_assistant/image_processor.py
"""
Extract and describe images from documents using Gemini Vision.
No OS-level dependencies required.
"""
import io
import base64
import logging
from decouple import config

logger = logging.getLogger(__name__)


def extract_images_from_pdf(file_path):
    """
    Extract images embedded in a PDF file.
    Returns list of {image_bytes, page, image_index} dicts.
    Uses pypdf's built-in extraction — no Poppler needed.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    extracted = []

    for page_num, page in enumerate(reader.pages):
        # pypdf stores images in /Resources/XObject
        if '/XObject' not in page.get('/Resources', {}):
            continue

        xobjects = page['/Resources']['/XObject'].get_object()

        for obj_name, obj_ref in xobjects.items():
            obj = obj_ref.get_object()

            if obj.get('/Subtype') != '/Image':
                continue

            try:
                # Get image data
                image_data = obj.get_data()
                width = obj.get('/Width', 0)
                height = obj.get('/Height', 0)

                # Skip tiny images (likely icons, bullets, decorations)
                if width < 100 or height < 100:
                    continue

                color_space = obj.get('/ColorSpace', '')
                bits = obj.get('/BitsPerComponent', 8)

                extracted.append({
                    'image_bytes': image_data,
                    'page': page_num + 1,
                    'image_index': len(extracted),
                    'width': width,
                    'height': height,
                    'color_space': str(color_space),
                })

                logger.info(f"Extracted image {len(extracted)} from page {page_num + 1}: {width}x{height}")

            except Exception as e:
                logger.warning(f"Could not extract image on page {page_num + 1}: {e}")
                continue

    return extracted


def describe_image_with_gemini(image_bytes, context=""):
    """
    Use Gemini Vision to generate a text description of an image.
    This converts visual content into searchable text.

    Why Gemini Vision instead of OCR?
    - OCR only reads text in images
    - Gemini Vision understands charts, diagrams, photos, tables
    - The description becomes a searchable text chunk in your RAG system
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config('GEMINI_API_KEY'))

    # Convert raw bytes to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Try to determine image format from magic bytes
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime_type = 'image/png'
    elif image_bytes[:2] == b'\xff\xd8':
        mime_type = 'image/jpeg'
    elif image_bytes[:4] == b'GIF8':
        mime_type = 'image/gif'
    else:
        mime_type = 'image/jpeg'  # default assumption

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                f"""Describe this image in detail for a document search system.
Include:
- What type of image this is (chart, diagram, photo, table, screenshot, etc.)
- All text visible in the image
- Key data points if it's a chart or graph
- Main subject if it's a photo
- Any numbers, dates, or measurements visible
Context from surrounding document: {context}

Be specific and thorough — this description will be used to answer questions."""
            ]
        )
        description = response.text
        logger.info(f"Generated image description: {description[:100]}...")
        return description

    except Exception as e:
        logger.error(f"Gemini Vision failed: {e}")
        return f"[Image on page — could not be described: {str(e)[:100]}]"


def process_pdf_images(file_path):
    """
    Extract all images from a PDF and generate text descriptions.
    Returns list of chunks ready for embedding.
    """
    images = extract_images_from_pdf(file_path)

    if not images:
        logger.info("No significant images found in document")
        return []

    image_chunks = []

    for img_data in images:
        description = describe_image_with_gemini(
            img_data['image_bytes'],
            context=f"Page {img_data['page']} of document"
        )

        image_chunks.append({
            'text': f"[IMAGE on page {img_data['page']}]\n{description}",
            'metadata': {
                'page': img_data['page'],
                'chunk_type': 'image',
                'image_dimensions': f"{img_data['width']}x{img_data['height']}",
                'content_type': 'image_description',
            }
        })

    logger.info(f"Processed {len(image_chunks)} images from PDF")
    return image_chunks