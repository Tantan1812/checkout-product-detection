import unittest

from checkout_vision.detection.detector import Detection, ProductDetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, classes, confidences):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences)


class FakeResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {2: "Candy"}
        self.annotated = object()

    def plot(self, **kwargs):
        self.plot_options = kwargs
        return self.annotated


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class ProductDetectorTests(unittest.TestCase):
    def test_predict_returns_box_class_confidence_and_annotated_frame(self):
        frame = object()
        result = FakeResult(FakeBoxes([[1, 2, 30, 40]], [2], [0.91]))
        model = FakeModel(result)
        detector = ProductDetector(
            "models/weights/best.pt",
            confidence=0.3,
            iou=0.5,
            image_size=640,
            device=0,
            model=model,
        )

        detections, annotated = detector.predict(frame)

        self.assertEqual(
            detections,
            [Detection(box=(1.0, 2.0, 30.0, 40.0), class_id=2, label="Candy", confidence=0.91)],
        )
        self.assertIs(annotated, result.annotated)
        self.assertEqual(
            model.calls[0],
            {
                "source": frame,
                "conf": 0.3,
                "iou": 0.5,
                "imgsz": 640,
                "device": 0,
                "verbose": False,
            },
        )

    def test_predict_returns_empty_list_when_no_boxes_exist(self):
        result = FakeResult(None)
        detector = ProductDetector("unused.pt", model=FakeModel(result))

        detections, annotated = detector.predict(object())

        self.assertEqual(detections, [])
        self.assertIs(annotated, result.annotated)

    def test_predict_maps_class_names_from_a_list(self):
        result = FakeResult(FakeBoxes([[0, 1, 2, 3]], [1], [0.7]), names=["Beans", "Cake"])
        detector = ProductDetector("unused.pt", model=FakeModel(result))

        detections, _ = detector.predict(object())

        self.assertEqual(detections[0].label, "Cake")


if __name__ == "__main__":
    unittest.main()
