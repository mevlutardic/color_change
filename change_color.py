import cv2
import os
import numpy as np
import threading


class GrabCut(threading.Thread):
    def __init__(self, windowName, srcfolderpath, dstfolderpath,kSize):
        threading.Thread.__init__(self)

        self.windowName = windowName
        self.srcfolderpath = srcfolderpath
        self.dstfolderpath = dstfolderpath
        self.kSize = kSize
        
        self.i = 0
        self.sourcelist = os.listdir(self.srcfolderpath)
        self.informations()
        cv2.namedWindow(self.windowName, cv2.WINDOW_AUTOSIZE)

        cv2.setMouseCallback(self.windowName, self.drawing_process)

        self.mode = False

        path = os.path.join(self.srcfolderpath, self.sourcelist[self.i])
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        self.resize_orj_img = cv2.resize(self.image, (int(self.image.shape[1] * self.kSize), int(self.image.shape[0] * self.kSize)))

        self.image_new = self.resize_orj_img.copy()

        self.mask = np.zeros(self.image_new.shape[:2], dtype=np.uint8)
        
        self.grapcut_mask = None

        self.output = np.zeros(self.image_new.shape, np.uint8)
        self.init_trakingBar()
        cv2.imshow(self.windowName, self.image_new)
     

        self.BLUE = [255, 0, 0]
        RED = [0, 0, 255]
        GREEN = [0, 255, 0]
        BLACK = [0, 0, 0]
        WHITE = [255, 255, 255]

        self.DRAW_BG = {'color': BLACK, 'val': 0}
        self.DRAW_FG = {'color': WHITE, 'val': 1}
        self.DRAW_PR_FG = {'color': GREEN, 'val': 3}
        self.DRAW_PR_BG = {'color': RED, 'val': 2}

        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rectangle = False
        self.rect_over = False
        self.value = self.DRAW_FG
        self.rect_or_mask = None
        self.start_point_rect = (0, 0)
        self.thickness = 3

        self.draw_circle = False
        self.current_image = None
        self.img = None
        self.finaldst =None
        
        self.update_photo(0)
        
    def apply_color_change(self,image, mask, diff_hue, brightness_change):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(image)

        hnew = np.mod(h + diff_hue, 180).astype(np.uint8)
        
        vnew = np.clip(v + brightness_change, 0, 255).astype(np.uint8)
        
        hsv_new = cv2.merge([hnew, s, vnew])
        bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        
        mask_3ch = cv2.merge([mask, mask, mask])
        
        result = np.where(mask_3ch == 255, bgr_new, self.image_new)
        
        return result
        
    def on_trackbar(self,val):
        self.thickness = cv2.getTrackbarPos("Thickness Size", self.windowName)
        red_hue = cv2.getTrackbarPos("Color Scale Hue Up", self.windowName)
        blue_hue = cv2.getTrackbarPos("Color Scale Hue Down", self.windowName)
        diff_hue = red_hue - blue_hue
        
        
       
        brightness_change = cv2.getTrackbarPos("Brightness", self.windowName) - 100
        
 
        try :
            result = self.apply_color_change(self.image_new, self.grapcut_mask, diff_hue, brightness_change)
            
            self.finaldst = result
            cv2.imshow(self.windowName, result)
        except:
            pass
        
       
        
    def init_trakingBar(self):
        cv2.createTrackbar("Thickness Size", self.windowName, 1, 10, self.on_trackbar)
        cv2.createTrackbar("Color Scale Hue Up", self.windowName, 0, 255, self.on_trackbar)
        cv2.createTrackbar("Color Scale Hue Down", self.windowName, 0, 255, self.on_trackbar)

        cv2.createTrackbar("Brightness", self.windowName, 0, 100, self.on_trackbar) 
        
        self.on_trackbar(0)
        
        
    def informations(self):
        print("""
                Keyboard Controls:
                ------------------
                Press 'q' : Exit the loop
                Press 'n' : Go to the next photo
                Press 'b' : Go to the previous photo
                Press 'm' : Toggle mode (e.g., exit drawing mode)
                Press 'c' : Enable circle drawing mode
                Press 'g' : Run the GrabCut algorithm
                Press 's' : Save results image
                
                Drawing Modes:
                --------------
                Press '0' : Draw background (BG)
                Press '1' : Draw foreground (FG)
                Press '2' : Draw probable background (PR_BG)
                Press '3' : Draw probable foreground (PR_FG)
            """)


    def run(self):
        while True:
            if not self.mode and not self.draw_circle:
                cv2.imshow(self.windowName, self.image_new)

            code = cv2.waitKey(1)

            if code == ord("q"):  # Press 'q' to exit the loop
                print(f'\r --- Exiting the program...                                      /n', end='', flush=True)
                break
            elif code == ord("n"):  # Press 'n' to go to the next photo
                if self.i < len(self.sourcelist) - 1:
                    self.i += 1
                    self.update_photo(self.i)
                    print(f'\r --- Moving to the next photo...                              ', end='', flush=True)
            elif code == ord("b"):  # Press 'b' to go to the previous photo
                if self.i > 0:
                    self.i -= 1
                    self.update_photo(self.i)
                    print(f'\r --- Moving to the previous photo...                          ', end='', flush=True)
            elif code == ord("s"): # Press 's' save the current photo
                dstpath = self.dstfolderpath+"/"+self.current_image+"_result"
                if self.finaldst is None:
                    print(f"{self.finaldst} is empty (None) \n", end='', flush=True)
                elif self.finaldst.size == 0:
                    print("Final destination image is empty (size=0) \n", end='', flush=True)
                else:  
                    cv2.imwrite(dstpath, self.finaldst)
                    print(f"Image saved to {dstpath} \n")
                    
            elif code == ord("m"):  # Press 'm' to toggle mode (e.g., exit drawing mode)
                self.mode = not self.mode
                self.draw_circle = False
                print(f'\r --- Toggling mode...                                     ', end='', flush=True)
            elif code == ord("c"):  # Press 'c' to enable circle drawing mode
                self.draw_circle = True
                print(f'\r -- Enable circle drawing mode...                         ', end='', flush=True)
            elif code == ord("g"):  # Press 'g' to run the GrabCut algorithm
                self.grab_cut()
                print(f'\r --- Running GrabCut algorithm...                         ', end='', flush=True)
            elif code == ord('0'):  # Press '0' for background (BG) drawing
                self.value = self.DRAW_BG 
                print(f'\r -- Background drawing selected...                        ', end='', flush=True)
            elif code == ord('1'):  # Press '1' for foreground (FG) drawing
                self.value = self.DRAW_FG
                print(f'\r -- Foreground drawing selected...                        ', end='', flush=True)
            elif code == ord('2'):  # Press '2' for probable background (PR_BG) drawing
                self.value = self.DRAW_PR_BG
                print(f'\r -- Probable background drawing selected...               ', end='', flush=True)
            elif code == ord('3'):  # Press '3' for probable foreground (PR_FG) drawing
                self.value = self.DRAW_PR_FG
                print(f'\r -- Probable foreground drawing selected...               ', end='', flush=True)


        cv2.destroyAllWindows()
            
    def grab_cut(self):
        if self.rect_or_mask == 0:  # grabcut with rect
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            self.img = cv2.cvtColor(self.image_new, cv2.COLOR_RGBA2RGB)
            cv2.grabCut(self.img, self.mask, self.rect, bgdmodel, fgdmodel, 7, cv2.GC_INIT_WITH_RECT)
            self.rect_or_mask = 1
        elif self.rect_or_mask == 1:  # grabcut with mask
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            self.img = cv2.cvtColor(self.image_new, cv2.COLOR_RGBA2RGB)
            cv2.grabCut(self.img, self.mask, self.rect, bgdmodel, fgdmodel, 3, cv2.GC_INIT_WITH_MASK)
        
        self.grapcut_mask = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
      
        
        if np.any(self.grapcut_mask): 
            overlay = self.image_drawing.copy()
            self.output = cv2.bitwise_and(self.img, self.img, mask=self.grapcut_mask)
            contours, hierarchy = cv2.findContours( cv2.cvtColor(self.output, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.image_drawing, contours, -1, (255, 100, 205), -1)
            alpha = 0.5
            self.image_drawing = cv2.addWeighted(overlay, alpha, self.image_drawing, 1 - alpha, 0) 
            cv2.imshow(self.windowName, self.image_drawing)
            self.image_drawing = self.img.copy()
        else:
            print("grapcut_mask is empty.")
    


    def update_photo(self, i):
        path = os.path.join(self.srcfolderpath, self.sourcelist[i])
        self.current_image = self.sourcelist[i]
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.image = cv2.resize(self.image, (int(self.image.shape[1] * self.kSize), int(self.image.shape[0] * self.kSize)))
        self.image_new = self.image.copy()
        self.mask = np.zeros(self.image_new.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.image_new.shape, np.uint8)

    def drawing_process(self, event, x, y, flags, param):
        if self.mode and not self.draw_circle:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rectangle = True
                self.start_point_rect = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectangle:
                    self.image_drawing = self.image_new.copy()
                    ix = self.start_point_rect[0]
                    iy = self.start_point_rect[1]
                    overlay = self.image_drawing.copy() 
                    cv2.rectangle(self.image_drawing, (ix, iy), (x, y), self.BLUE, 3)
                    alpha = 0.5
                    self.image_drawing = cv2.addWeighted(overlay, alpha, self.image_drawing, 1 - alpha, 0) 
                    self.rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
                    self.rect_or_mask = 0
                    cv2.imshow(self.windowName, self.image_drawing)
            elif event == cv2.EVENT_LBUTTONUP:
                self.rectangle = False
                self.rect_over = True
                ix = self.start_point_rect[0]
                iy = self.start_point_rect[1]
                self.rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
                self.rect_or_mask = 0

        elif self.rect_over and self.draw_circle:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                cv2.circle(self.image_drawing, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    overlay = self.image_drawing.copy()
                    cv2.circle(self.image_drawing, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
                    alpha = 0.5
                    self.image_drawing = cv2.addWeighted(overlay, alpha, self.image_drawing, 1 - alpha, 0) 
                    cv2.imshow(self.windowName, self.image_drawing)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing:
                    self.drawing = False
                    overlay = self.image_drawing.copy()
                    cv2.circle(self.image_drawing, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
                    alpha = 0.5
                    self.image_drawing = cv2.addWeighted(overlay, alpha, self.image_drawing, 1 - alpha, 0) 
                    cv2.imshow(self.windowName, self.image_drawing)
                   

if __name__ == '__main__':
    src_image_folder = "./images"
    dst_image_folder = "./results"
    
    kSize = 1
    thread = GrabCut("Window", src_image_folder,dst_image_folder, kSize)
    thread.run()