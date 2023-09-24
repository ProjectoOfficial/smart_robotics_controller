import os
import cv2
import numpy as np
from PIL import Image

class ArucoDetector:
    
    def __init__(self):
        pass
    
    def loadCustomArucoLib(self):
        '''
        @Description: La funzione loadCustomArucoLib carica la libreria custom.
        '''
        scripts_dir = os.path.dirname(os.path.dirname(__file__))
        database_dir = os.path.join(scripts_dir, "data", "aruco_database")
        
        lib = []
        for i in range(15):
            data_matrix = np.loadtxt(os.path.join(database_dir, 'landmark-' + str(i+1) + '.txt')) 
            lib.append(data_matrix)

        return lib


    def box2xyxy(self, box: tuple) -> tuple:
        x, y, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        return (x, y, x2, y2)  


    def findSquare(self, frame):
        '''
        @Description: La funzione findSquare permette di trovare i contorni nell'immagine .

        @param: frame - immagine di input
        @param: cands - lista dei possibili candidati che: presentano 4 vertici, hanno un'area compresa tra due valori in modo da 
        ridurre i candidati e sono convessi
        '''
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        img_area = frame.shape[0] * frame.shape[1]
        cands = []

        for cnt in contours:           
            box = cv2.boundingRect(cnt)
            aspect_ratio = box[3] /box[2]
            if aspect_ratio < 0.8:
                continue
             
            area = cv2.contourArea(cnt)
            if not (area > img_area * 0.0001) or not (area < img_area * 0.9):
                continue
            
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4:
                continue            
            
            cands.append(cnt)

        return cands


    def PerspectiveTransform(self, img):
        '''
        @Description: La funzione PerspectiveTransform permette di trovare i vertici di un candidato aruco in modo da 
        migliorare la sua prospettiva nei passaggi successivi all'interno del main

        @param: img - immagine di input
        @param: corners - lista dei possibili candidati che: presentano 4 vertici, hanno un'area minore di quella dell'immagine data in input
        '''
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        img_area = frame.shape[0] * frame.shape[1]
        corners = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if (len(approx) == 4 and area < img_area):
                corners.append(approx)

        return corners


    def drawGrid(self, img, x, y, w, h, latoBit):
        '''
        @Description: La funzione drawGrid disegna una griglia 9x9 sull'aruco.

        @param: img - immagine di input
        @param: x,y,w,h - coordinate dei punti di interesse dell'aruco
        @param: latoBit - variabile di appoggio che serve per avere la giusta spaziatura tra le righe e colonne della griglia
        '''
        for i in range(10):
            cv2.line(img, (x + latoBit, y), (x + latoBit, y + h), (0, 255, 255), thickness=2)
            latoBit = latoBit + 64

        latoBit = 0
        for i in range(10):
            cv2.line(img, (x, y + latoBit), (x + w, y + latoBit), (0, 255, 255), thickness=2)
            latoBit = latoBit + 64


    def makeBooleanMatrix(self, marker, box_centreX, box_centreY):
        '''
        @Description: La funzione makeBooleanMatrix costruisce la matrice booleana (0=black, 1=white) dell'aruco.

        @param: marker - immagine di input
        @param: box_centerX, box_centerY - definiscono la posizione del primo centro del primo quadratino creato dalla griglia
        '''
        latoBit = 0
        image_rgb = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_rgb, 150, 255, cv2.THRESH_BINARY)
        image_rgb = Image.fromarray(thresh)
        matrice = np.empty((9, 9))

        for i in range(9):
            for j in range(9):
                rgb_value = image_rgb.getpixel((box_centreX + latoBit, box_centreY))
                latoBit = latoBit + 64
                if rgb_value == 0:  # 0 corrisponde al valore nero
                    matrice[i][j] = 0

                else:
                    matrice[i][j] = 1

                if j == 8:
                    latoBit = 0
                    box_centreY = box_centreY + 64

        return matrice


    def compareMatrix(self, matrix, lib):
        '''
        @Description: La funzione compareMatrix permette di comparare la matrice costruita dalla funzione precedente, con quelle
        presenti all'interno della libreria.

        @param: matrix - matrice booleana del candidato
        @param: lib - libreria che comprende le matrici dei 15 aruco
        @param: sig - variabile che definisce il numero dell'aruco analizzato
        '''
        sign = 0
        for i in range(15):
            equal_arrays = (matrix == lib[i]).all()
            if equal_arrays:
                sign = i + 1

        if sign == 0:
            return None
        else:
            return sign
        
        
    def process(self, l_image:np.ndarray, r_image:np.ndarray, depth:np.ndarray, debug:bool=False):
        """
        @Description: questa funzione prende in input un immagine e utilizza i metodi di questa classe per trovare i landmark e classificarli
        
        @param: img - immagine in input (np.ndarray)
        """
        library = self.loadCustomArucoLib()
        
        l_img = cv2.cvtColor(l_image.copy(), cv2.COLOR_RGB2BGR)
        bbox = self.findSquare(l_img)

        shape = (573, 573)
        detections = []
        for box in bbox:
            bbox = cv2.boundingRect(box)
            x, y, w, h = bbox
            
            if debug:
                l_img = cv2.rectangle(l_img, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.imshow("l",l_img)
                cv2.waitKey(1000//60)

            crop = l_image[y:y + h, x:x + w]
            crop = cv2.resize(crop, shape)
            corners = self.PerspectiveTransform(crop)
            
            for corner in corners:
                cornersIniziali = np.float32(corner)
                cornersFinali = np.float32([[0, 0], [573, 0], [573, 573], [0, 573]])
                M = cv2.getPerspectiveTransform(cornersIniziali, cornersFinali)
                wrapped = cv2.warpPerspective(crop, M, shape)

                # drawGrid
                vert_x, vert_y, latoBit = (0, 0, 0)
                width, height = shape
                self.drawGrid(wrapped, vert_x, vert_y, width, height, latoBit)

                # CreateBooleanMatrix (0 = Black, 1 = White)
                box_centreX = int(vert_x + 32)
                box_centreY = int(vert_y + 32)
                matrix = self.makeBooleanMatrix(wrapped, box_centreX, box_centreY)

                # Compare matrix for determinate signature of marker
                signature = self.compareMatrix(matrix, library)
                if signature != None:
                    detections.append([signature, (x, y, w, h)])
                    print(str(signature) + ' x1:' + str(x) + ' y1:' + str(y) + ' x2:' + str(x + w) + ' y2:' + str(y + h))
                    cv2.rectangle(l_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(l_image, str(signature), (int(x + w/2), y - 4), cv2.FONT_ITALIC, 1, (255, 0, 0), thickness=2)
                    
        if debug:
            print(detections)
        return l_image, detections
        