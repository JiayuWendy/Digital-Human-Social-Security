import os
import urllib.request
import zipfile
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ✅ 字体下载与解压
def download_and_extract_font(zip_url, save_dir):
    """
    下载并解压字体压缩包
    :param zip_url: 字体压缩包下载链接
    :param save_dir: 解压后的保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    zip_path = os.path.join(save_dir, "font.zip")

    try:
        print(f"⬇️ 正在下载字体压缩包: {zip_url}")
        urllib.request.urlretrieve(zip_url, zip_path)
        print(f"✅ 下载完成: {zip_path}")

        # 解压字体包
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)

        print(f"✅ 字体解压完成: {save_dir}")

        # 删除压缩包
        os.remove(zip_path)

    except Exception as e:
        print(f"❌ 下载或解压字体失败: {e}")


# ✅ 自动选择字体
def get_font(font_dir="fonts", font_size=24):
    """
    自动加载字体：
    - 检测字体文件是否存在
    - 如果不存在，则自动下载并解压
    """
    font_name = "msyh.ttc"  # 中文字体名称
    font_path = os.path.join(font_dir, font_name)

    # 检测字体是否存在
    if os.path.exists(font_path):
        print(f"✅ 使用本地字体: {font_path}")
        return ImageFont.truetype(font_path, font_size)

    # 字体项目压缩包地址（使用 GitHub 项目）
    font_zip_url = "https://github.com/CroesusSo/msyh/archive/refs/heads/main.zip"

    print("⚠️ 字体文件不存在，尝试自动下载...")

    # 下载并解压字体
    download_and_extract_font(font_zip_url, font_dir)

    # 解压后的目录结构修复
    extracted_dir = os.path.join(font_dir, "msyh-main")

    # 将字体文件移动到 fonts 目录
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".ttc") or file.endswith(".ttf") or file.endswith(".otf"):
                shutil.move(os.path.join(root, file), os.path.join(font_dir, file))
                print(f"✅ 移动字体: {file}")

    # 删除临时解压文件夹
    shutil.rmtree(extracted_dir)

    # 检测字体是否成功下载
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)
    else:
        print("❌ 字体加载失败，使用默认字体")
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
