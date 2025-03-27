"""font.py"""
import os
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ✅ 字体下载函数
def download_font(font_url, save_path):
    """
    下载字体文件
    :param font_url: 下载链接
    :param save_path: 保存路径
    """
    try:
        print(f"⬇️ 正在下载字体: {font_url}")
        urllib.request.urlretrieve(font_url, save_path)
        print(f"✅ 字体下载成功: {save_path}")
        return True
    except Exception as e:
        print(f"❌ 下载字体失败: {e}")
        return False


# ✅ 自动选择字体
def get_font(font_path="msyh.ttc", font_size=24):
    """
    自动加载字体：
    - 优先使用指定字体路径
    - 如果不存在，则自动下载
    - 离线环境提示手动下载
    """
    # 检测字体文件是否存在
    if os.path.exists(font_path):
        print(f"✅ 使用本地字体: {font_path}")
        return ImageFont.truetype(font_path, font_size)

    # 字体存储路径
    fonts_dir = os.path.join(os.getcwd(), "fonts")
    os.makedirs(fonts_dir, exist_ok=True)

    # 字体文件路径
    font_file = os.path.join(fonts_dir, "msyh.ttc")

    # ✅ 下载字体
    font_urls = [
        "https://github.com/google/fonts/raw/main/apache/noto/NotoSansSC-Regular.otf",  # 开源字体
        "https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/SourceHanSansSC-Regular.otf"
    ]

    # 如果本地字体不存在，则自动下载
    if not os.path.exists(font_file):
        print("⚠️ 未找到字体文件，尝试自动下载...")

        # 尝试下载字体
        success = False
        for url in font_urls:
            if download_font(url, font_file):
                success = True
                break

        if not success:
            print("❌ 无法自动下载字体，请手动下载并放置到 fonts 目录下！")
            return ImageFont.load_default()

    # 加载下载的字体
    try:
        return ImageFont.truetype(font_file, font_size)
    except IOError:
        print("❌ 加载字体失败，使用默认字体")
        return ImageFont.load_default()


# ✅ 中文显示函数
def put_chinese_text(frame, text, position, font_size=24, color=(0, 255, 0)):
    """
    在 OpenCV 窗口上绘制中文文本
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 自动选择字体
    font = get_font(font_size=font_size)

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ✅ 测试
if __name__ == "__main__":
    # 测试绘制中文文本
    frame = np.zeros((500, 800, 3), dtype=np.uint8)

    frame = put_chinese_text(frame, "测试自动下载字体功能", (20, 50), font_size=30, color=(255, 0, 0))
    frame = put_chinese_text(frame, "请检查 fonts 文件夹是否有下载的字体", (20, 100), font_size=30, color=(0, 255, 0))

    cv2.imshow("中文字体测试", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
