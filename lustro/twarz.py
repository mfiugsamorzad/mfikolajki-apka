# -*- coding: utf-8 -*-
"""
Lokalizacja twarzy i wyznaczanie miejsca na czapkę.

Podstawą jest MediaPipe Face Detection (BlazeFace): szybki (~2–4 ms na CPU),
zwraca prostokąt twarzy oraz 6 punktów kluczowych, w tym ŚRODKI OCZU. Oczy są
dla nas kluczowe:

  * odległość między oczami (d) to stabilna miara skali głowy – lepsza niż
    sam prostokąt, który "pływa" między klatkami,
  * nachylenie odcinka oko–oko to przechylenie głowy (roll) – czapka obraca się
    razem z głową,
  * czoło leży na prostej prostopadłej do linii oczu, ok. 0,85·d nad nią –
    tam kotwiczymy dolną krawędź (otok) czapki.

Gdy MediaPipe Face Detection jest niedostępne, używamy Haar Cascade (prostokąt
twarzy, położenie oczu szacowane z proporcji twarzy, bez przechylenia).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    box: tuple            # (x, y, w, h) w pikselach
    eye_left: np.ndarray  # oko po LEWEJ stronie ekranu (po odbiciu lustrzanym)
    eye_right: np.ndarray


@dataclass
class HeadPose:
    forehead: np.ndarray  # punkt na czole, w którym ląduje otok czapki
    face_width: float     # szerokość głowy w pikselach
    angle: float          # przechylenie głowy w stopniach (dodatnie = zgodnie z ruchem wskazówek)


def head_pose(face: Face, forehead_up: float) -> HeadPose:
    el, er = face.eye_left, face.eye_right
    v = er - el
    d = max(float(np.hypot(*v)), 1.0)
    theta = math.atan2(v[1], v[0])  # oś y w obrazie rośnie w dół -> dodatni kąt = przechył w prawo
    up = np.array([math.sin(theta), -math.cos(theta)])  # prostopadła do linii oczu, "w górę głowy"
    eyes_center = (el + er) / 2
    forehead = eyes_center + up * (forehead_up * d)
    # Szerokość głowy: prostokąt z detektora, ale nie mniej niż 2,2 rozstawu oczu
    # (prostokąt bywa za wąski przy twarzy lekko odwróconej od kamery).
    face_w = max(float(face.box[2]), 2.2 * d)
    return HeadPose(forehead, face_w, math.degrees(theta))


class FaceLocator:
    """MediaPipe Face Detection z zapasowym Haar Cascade."""

    def __init__(self, mp_face_detection=None, haar_dirs=(), model_selection: int = 0,
                 min_confidence: float = 0.5):
        self.detector = None
        self.cascade = None
        if mp_face_detection is not None:
            try:
                # model_selection=0: model krótkiego zasięgu (do ~2 m) – idealny dla lustra
                self.detector = mp_face_detection.FaceDetection(
                    model_selection=model_selection, min_detection_confidence=min_confidence)
            except Exception as exc:  # pragma: no cover - zależy od instalacji
                print(f"[uwaga] MediaPipe Face Detection niedostępne ({exc}).")
        if self.detector is None:
            name = "haarcascade_frontalface_default.xml"
            for d in haar_dirs:
                path = os.path.join(d, name)
                if os.path.isfile(path):
                    c = cv2.CascadeClassifier(path)
                    if not c.empty():
                        self.cascade = c
                        print("[info] Detekcja twarzy: Haar Cascade.")
                        break
        if self.detector is not None:
            print("[info] Detekcja twarzy: MediaPipe Face Detection.")
        elif self.cascade is None:
            print("[uwaga] Brak detektora twarzy – czapka będzie nakładana nad palcami.")

    @property
    def available(self) -> bool:
        return self.detector is not None or self.cascade is not None

    def locate(self, rgb: np.ndarray, bgr: np.ndarray) -> list:
        h, w = bgr.shape[:2]
        faces = []
        if self.detector is not None:
            res = self.detector.process(rgb)
            for det in res.detections or []:
                ld = det.location_data
                bb = ld.relative_bounding_box
                box = (bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h)
                kp = ld.relative_keypoints  # 0, 1 = oczy; 2 = nos; 3 = usta; 4, 5 = uszy
                e1 = np.array([kp[0].x * w, kp[0].y * h], np.float32)
                e2 = np.array([kp[1].x * w, kp[1].y * h], np.float32)
                el, er = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
                faces.append(Face(box, el, er))
        elif self.cascade is not None:
            small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            for (x, y, fw, fh) in self.cascade.detectMultiScale(gray, 1.15, 6, minSize=(40, 40)):
                x, y, fw, fh = (float(v) * 2 for v in (x, y, fw, fh))
                # Oczy w typowych proporcjach twarzy: ~38% wysokości, 30% / 70% szerokości
                ey = y + 0.38 * fh
                faces.append(Face((x, y, fw, fh), np.array([x + 0.3 * fw, ey], np.float32),
                                  np.array([x + 0.7 * fw, ey], np.float32)))
        return faces

    def close(self):
        if self.detector is not None:
            self.detector.close()
