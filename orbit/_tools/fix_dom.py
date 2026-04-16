import re

path = r'C:\Users\aadya\Desktop\projects\computer-use\orbit\.claude\worktrees\hopeful-engelbart\orbit\_tools\playwright_tools.py'
with open(path, 'r') as f:
    c = f.read()

c = c.replace('from typing import Dict, Any', 'from typing import Dict, Any\nfrom urllib.parse import urlparse', 1)

c = c.replace('async def dom_list_frames() -> list[Dict[str, Any]]:', 'async def dom_list_frames() -> Dict[str, Any]:', 1)
c = c.replace('        return []\n', '        return {"status": "error", "message": "Browser is not active."}\n', 1)
c = c.replace('        for f in frames:\n            if src and src in f["url"]:\n                f["selector"] = f"iframe[src*=\'{src.split(\'?\')[0]}\']"\n    return frames\n', '        for f in frames:\n            if src and urlparse(src).path == urlparse(f["url"]).path:\n                f["selector"] = f"iframe[src*=\'{src.split(\'?\')[0]}\']"\n    return {"status": "success", "frames": frames}\n', 1)

old_click = """    frame = global_browser.active_frame_or_page
    try:
        await frame.get_by_text(text, exact=False).first.click(timeout=5000)
        return {"status": "success", "message": f"Clicked element with text '{text}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}"""

new_click = """    frame = global_browser.active_frame_or_page
    try:
        try:
            await frame.get_by_role("button", name=text, exact=False).first.click(timeout=3000)
        except Exception:
            await frame.get_by_text(text, exact=False).first.click(timeout=3000)
        return {"status": "success", "message": f"Clicked element with text '{text}'"}
    except Exception as e:
        return {"status": "error", "message": str(e)}"""

c = c.replace(old_click, new_click, 1)

with open(path, 'w') as f:
    f.write(c)
