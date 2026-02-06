from PIL import Image
from PIL.ExifTags import TAGS

def scan_metadata(image_path):
    """
    Extracts software signatures and camera info to find editing traces.
    """
    try:
        img = Image.open(image_path)
        info = img.getexif()
        report = {}
        
        if not info:
            return None, "No EXIF metadata found. (Common for screenshots or social media images)"

        for tag_id in info:
            tag = TAGS.get(tag_id, tag_id)
            data = info.get(tag_id)
            report[tag] = data
            
        # Common signatures of editing software
        editing_tools = ['photoshop', 'gimp', 'pixlr', 'adobe', 'canva', 'picsart', 'snapseed']
        software = str(report.get('Software', '')).lower()
        
        for tool in editing_tools:
            if tool in software:
                return report, f"⚠️ ALERT: Software trace detected ({software.upper()})"
                
        return report, "✅ Metadata looks clean. No suspicious software found."
    except Exception as e:
        return None, f"Error: {e}"