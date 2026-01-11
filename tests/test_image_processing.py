import io
import unittest

from PIL import Image

from image_utils import _compress_bytes_to_limit


class ImageProcessingTests(unittest.TestCase):
    def test_transparent_png_composites_on_white_when_forced_jpeg(self) -> None:
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        img.putpixel((5, 5), (255, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        ok, out_bytes, ct, err = _compress_bytes_to_limit(
            buf.getvalue(),
            max_mb=1,
            _purpose="test",
            prefer_fmt="JPEG",
        )

        self.assertTrue(ok, msg=err)
        self.assertEqual(ct, "image/jpeg")

        out_img = Image.open(io.BytesIO(out_bytes))
        self.assertEqual(out_img.mode, "RGB")
        pixel = out_img.getpixel((0, 0))
        self.assertTrue(all(channel >= 250 for channel in pixel), msg=f"Pixel was not near white: {pixel}")


if __name__ == "__main__":
    unittest.main()
