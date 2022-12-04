## Confusing Image Quality Assessment: Towards Better Augmented Reality Experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet


import csv
import os
from PIL import Image, ImageTk
import pandas as pd
import random
import scipy.io as scio
import tkinter.filedialog
import tkinter as tk 
import tkinter.messagebox 


# shuffle = False
shuffle = True


class Browser():
    def __init__(self, window):

        self.path = None
        self.path_display = tk.StringVar()
        self.path_display2 = tk.StringVar()
        self.path_display3 = tk.StringVar()
        self.subject_name_display = tk.StringVar()
        self.start_number_display = tk.IntVar()
        self.subject_name = ''
        self.imgs_name = None
        self.imgs_id = None
        self.canvas = None
        self.number = 0
        self.display_number = tk.StringVar()
        self.current_img = None
        self.current_tk_img = None

        self.shuffle = shuffle

        # score
        self.current_score = 0
        self.scores = []
        self.current_score2 = 0
        self.scores2 = []

        # image window definition
        self.img_window_h = 620
        self.img_window_w = 1600

        # image definition
        self.img_h = None
        self.img_w = None
        self.img_c = None

        self.window = window
        
        self.define_window_property()
        self.define_frames()
        self.define_frame1()
        self.define_frame2()
        
    # -------------------------------------------------------------------------------------------------------------------
    #                                               interface below
    # -------------------------------------------------------------------------------------------------------------------
    # define window properties
    def define_window_property(self):
        self.window.title('Image Quality Assessment')
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.window_width = 1920
        self.window_height = 1080
        x = (self.screen_width-self.window_width) / 2
        y = (self.screen_height-self.window_height) / 2
        self.window.geometry("%dx%d+%d+%d" % (self.window_width, self.window_height, x, y)) # x,y put the window at the center of the screen
        # define background color
        # bg = tk.Label(self.window, bg='#3c3f41')
        # bg.place(height=self.window_height, width=self.window_width, x=0, y=0)
    
    # define window frame properties
    def define_frames(self):
        # bg: ['red','blue','yellow','green','white','black']
        # below are test mode:
        # self.frame1 = tk.Frame(self.window, height=720, width=1080, bg='blue')
        # self.frame1.pack(side='left')
        # self.frame2 = tk.Frame(self.window, height=960, width=400, bg='green')
        # self.frame2.pack(side='right')
        self.frame1 = tk.Frame(self.window, height=self.img_window_h, width=self.img_window_w)
        self.frame1.place(x=160,y=20,anchor='nw')
        # self.frame2 = tk.Frame(self.window, height=300, width=1600, bg='green')
        self.frame2 = tk.Frame(self.window, height=300, width=1600)
        self.frame2.place(x=160,y=640,anchor='nw')

    # frame1: image frame
    def define_frame1(self):
        # self.canvas = tk.Canvas(self.frame1, bg='red', height=512, width=512) # test mode
        # image middle
        self.canvas = tk.Canvas(self.frame1, height=520, width=520)
        self.canvas.place(relx=0.5,rely=0.5,anchor='center')
        # image left
        self.canvas2 = tk.Canvas(self.frame1, height=520, width=520)
        self.canvas2.place(relx=0.165,rely=0.5,anchor='center')
        # image right
        self.canvas3 = tk.Canvas(self.frame1, height=520, width=520)
        self.canvas3.place(relx=0.835,rely=0.5,anchor='center')

    # frame2: display frame: define labels (instruction) in frame (window)
    def define_frame2(self):
        self.path = os.getcwd()
        # path
        tk.Label(self.frame2,text = "mixed path:").place(relx=0.05, rely=0.3, anchor='center')#.pack()#.place(x=0.0*self.window_width, y=0.025*self.window_height, anchor='w')
        tk.Entry(self.frame2, textvariable=self.path_display).place(relx=0.125, rely=0.3, anchor='center')
        tk.Button(self.frame2, text = "choose", command = self.select_path).place(relx=0.2, rely=0.3, anchor='center')
        tk.Label(self.frame2,text = "A path:").place(relx=0.05, rely=0.4, anchor='center')#.pack()#.place(x=0.0*self.window_width, y=0.025*self.window_height, anchor='w')
        tk.Entry(self.frame2, textvariable=self.path_display2).place(relx=0.125, rely=0.4, anchor='center')
        tk.Button(self.frame2, text = "choose", command = self.select_path2).place(relx=0.2, rely=0.4, anchor='center')
        tk.Label(self.frame2,text = "B path:").place(relx=0.05, rely=0.5, anchor='center')#.pack()#.place(x=0.0*self.window_width, y=0.025*self.window_height, anchor='w')
        tk.Entry(self.frame2, textvariable=self.path_display3).place(relx=0.125, rely=0.5, anchor='center')
        tk.Button(self.frame2, text = "choose", command = self.select_path3).place(relx=0.2, rely=0.5, anchor='center')
        # subject name
        tk.Label(self.frame2,text = "subject name:").place(relx=0.05, rely=0.7, anchor='center')#.pack()#.place(x=0.0*self.window_width, y=0.025*self.window_height, anchor='w')
        tk.Entry(self.frame2, textvariable=self.subject_name_display).place(relx=0.125, rely=0.7, anchor='center')
        self.button_subject = tk.Button(self.frame2, text = "enter", command = self.get_subject_name).place(relx=0.2, rely=0.7, anchor='center')
        # start number
        tk.Label(self.frame2,text = "start number:").place(relx=0.05, rely=0.9, anchor='center')#.pack()#.place(x=0.0*self.window_width, y=0.025*self.window_height, anchor='w')
        tk.Entry(self.frame2, textvariable=self.start_number_display).place(relx=0.125, rely=0.9, anchor='center')
        self.button_subject = tk.Button(self.frame2, text = "enter", command = self.get_start_number).place(relx=0.2, rely=0.9, anchor='center')
        # four button
        self.button_previous = tk.Button(self.frame2, text = "Prev", height=5, width=15, font=12, command = self.previous)
        self.button_next = tk.Button(self.frame2, text = "Next", height=5, width=15, font=12, command = self.next)
        self.button_start = tk.Button(self.frame2, text = "Start", height=5, width=15, font=12, command = self.start)
        self.button_finish = tk.Button(self.frame2, text = "Finish", height=5, width=15, font=12, command = self.finish)
        # choose scale (quality number)
        tk.Scale(self.frame2, label='score', from_=0, to=5, orient=tk.HORIZONTAL, length=300, showvalue=1, tickinterval=1, resolution=0.01, command=self.pass_score).place(relx=0.38, rely=0.13, anchor='center')
        tk.Scale(self.frame2, label='score', from_=0, to=5, orient=tk.HORIZONTAL, length=300, showvalue=1, tickinterval=1, resolution=0.01, command=self.pass_score2).place(relx=0.62, rely=0.13, anchor='center')
        # display current number
        tk.Entry(self.frame2, textvariable=self.display_number).place(relx=0.9, rely=0.3, anchor='center')
        # shuffle button
        # self.button_shuffle = tk.Button(self.frame2, text = "Disable Shuffle", command = self.disable_shuffle)

        self.button_previous.place(relx=0.33, rely=0.5, anchor='center')
        self.button_next.place(relx=0.66, rely=0.5, anchor='center')
        self.button_start.place(relx=0.33, rely=0.85, anchor='center')
        self.button_finish.place(relx=0.66, rely=0.85, anchor='center')
        # self.button_shuffle.place(relx=0.05, rely=0.1, anchor='center')
        self.button_previous["state"] = "disabled"
        self.button_next["state"] = "disabled"
        self.button_finish["state"] = "disabled"

    # -------------------------------------------------------------------------------------------------------------------
    #                                               functions below
    # -------------------------------------------------------------------------------------------------------------------
    # shuffle image
    def disable_shuffle(self):
        self.shuffle = False
        self.button_shuffle["state"] = "disabled"
    # select folder which contains test images
    def select_path(self):
        self.path = tkinter.filedialog.askdirectory()
        self.path_display.set(self.path)
        self.get_imgs_name()
    # ***********************************************************************
    # *** important: images name in path, path2 and path3 should be same. ***
    # ***********************************************************************
    def select_path2(self):
        self.path2 = tkinter.filedialog.askdirectory()
        self.path_display2.set(self.path2)
        # self.get_imgs_name2()
    def select_path3(self):
        self.path3 = tkinter.filedialog.askdirectory()
        self.path_display3.set(self.path3)
        # self.get_imgs_name3()

    # get all images' name in the selected folder
    def get_imgs_name(self):
        self.imgs_name =[]
        self.imgs_id = []
        if(self.path != '' and self.path != ()):
            dirs = os.listdir(self.path)
            cnt = 0
            for dir in dirs:
                if dir.endswith(".jpg") or dir.endswith(".png") or dir.endswith(".bmp"):
                    # print(dir)
                    self.imgs_name.append(dir)
                    self.imgs_id.append(cnt)
                    cnt = cnt+1
        else:
            self.imgs_name = []
        # shuffle
        if self.shuffle:
            mapIndexPosition = list(zip(self.imgs_name, self.imgs_id))
            random.shuffle(mapIndexPosition)
            self.imgs_name, self.imgs_id = zip(*mapIndexPosition)


    # input subject name and enter
    def get_subject_name(self):
        self.subject_name = self.subject_name_display.get()
        print(self.subject_name)
        # continue last experiment
        if os.path.isfile(self.subject_name+"_seq.csv"):
            self.imgs_id_new = []
            with open(self.subject_name+"_seq.csv", newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.imgs_id_new.append(int(row[0]))
            dictionary = zip(self.imgs_id,self.imgs_name)
            dictionary_sorted = sorted(dictionary, key=lambda pair: self.imgs_id_new.index(pair[0]))
            self.imgs_id,self.imgs_name = zip(*dictionary_sorted)
        # a new subject
        else:
            for img_id in self.imgs_id:
                each_row = [img_id]
                with open(self.subject_name+"_seq.csv","a",newline='') as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow(each_row)

    # input subject name and enter
    def get_start_number(self):
        self.number = self.start_number_display.get()
        print(self.number)

    def previous(self):
        self.number -= 1
        if self.number < 0:
            self.button_previous["state"] = "disabled"
            self.number = 0
        self.pop_out_score()
        self.show_img()
        self.button_next["state"] = "normal"


    def next(self):
        self.append_and_save_score()
        if self.number>=len(self.imgs_name)-1:
            self.number += 1
            self.button_previous["state"] = "normal"
            self.button_next["state"] = "disabled"
            self.button_start["state"] = "disabled"
            self.button_finish["state"] = "normal"
        else:
            self.button_previous["state"] = "normal"
            self.number += 1
            self.show_img()


    def start(self):
        self.show_img()
        self.button_previous["state"] = "normal"
        self.button_next["state"] = "normal"
        self.button_start["state"] = "disabled"
        self.button_finish["state"] = "disabled"

    def finish(self):
        self.save_score()
        tkinter.messagebox.showwarning(message='Saved !!! Thank you !!!')  

    def show_img(self):
        self.display_number.set(self.number)
        print(os.path.join(self.path,self.imgs_name[self.number]))
        # image
        self.current_img = Image.open(os.path.join(self.path,self.imgs_name[self.number]))
        self.img_w, self.img_h = self.current_img.size
        if self.img_h>self.img_window_h or self.img_w>self.img_window_w:
            ratio = max(self.img_h/self.img_window_h,self.img_w/self.img_window_w)
            self.current_img.resize((int(self.img_h/ratio),int(self.img_w/ratio)))
        self.current_tk_img = ImageTk.PhotoImage(self.current_img)
        self.canvas.create_image(int(512/2),int(512/2),anchor='center',image=self.current_tk_img)
        # image 2
        self.current_img2 = Image.open(os.path.join(self.path2,self.imgs_name[self.number]))
        self.img_w, self.img_h = self.current_img2.size
        if self.img_h>self.img_window_h or self.img_w>self.img_window_w:
            ratio = max(self.img_h/self.img_window_h,self.img_w/self.img_window_w)
            self.current_img2.resize((int(self.img_h/ratio),int(self.img_w/ratio)))
        self.current_tk_img2 = ImageTk.PhotoImage(self.current_img2)
        self.canvas2.create_image(int(512/2),int(512/2),anchor='center',image=self.current_tk_img2)
        # image 3
        self.current_img3 = Image.open(os.path.join(self.path3,self.imgs_name[self.number]))
        self.img_w, self.img_h = self.current_img3.size
        if self.img_h>self.img_window_h or self.img_w>self.img_window_w:
            ratio = max(self.img_h/self.img_window_h,self.img_w/self.img_window_w)
            self.current_img3.resize((int(self.img_h/ratio),int(self.img_w/ratio)))
        self.current_tk_img3 = ImageTk.PhotoImage(self.current_img3)
        self.canvas3.create_image(int(512/2),int(512/2),anchor='center',image=self.current_tk_img3)

    # pass current selected score to the variable self.current_score
    def pass_score(self,v):
        self.current_score = v
    # pass current selected score to the variable self.current_score
    def pass_score2(self,v):
        self.current_score2 = v

    # pop out score
    def pop_out_score(self):
        self.scores.pop()
        self.scores2.pop()

    # append and save current results
    def append_and_save_score(self):
        self.scores.append(self.current_score)
        self.scores2.append(self.current_score2)
        each_row = [self.number,self.imgs_id[self.number],self.imgs_name[self.number],self.current_score,self.current_score2]
        with open(self.subject_name+"_temp.csv","a",newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(each_row)

    # save scores
    def save_score(self):
        scio.savemat(self.subject_name+'_scores.mat',{'scores':self.scores,'scores2':self.scores2})

        idx = 0
        whole_dataframe = []
        column_dataframe = ['id','img_id','img_name','score','score2']
        for img_id,img_name,score,score2 in zip(self.imgs_id,self.imgs_name,self.scores,self.scores2):
            idx += 1
            each_row = []
            each_row.append(idx)
            each_row.append(img_id)
            each_row.append(img_name)
            each_row.append(score)
            each_row.append(score2)
            whole_dataframe.append(each_row)
        whole_df = pd.DataFrame(whole_dataframe,columns = column_dataframe)
        whole_df.to_csv(self.subject_name+'_scores.csv',header=False,index=False)

        # sorted the order
        dictionary = zip(self.imgs_id,self.imgs_name,self.scores,self.scores2)
        dictionary_sorted = sorted(dictionary)
        self.imgs_id_sorted,self.imgs_name_sorted,self.scores_sorted,self.scores_sorted2 = zip(*dictionary_sorted)
        idx = 0
        whole_dataframe = []
        column_dataframe = ['id','img_id','img_name','score','score2']
        for img_id,img_name,score,score2 in zip(self.imgs_id_sorted,self.imgs_name_sorted,self.scores_sorted,self.scores_sorted2):
            idx += 1
            each_row = []
            each_row.append(idx)
            each_row.append(img_id)
            each_row.append(img_name)
            each_row.append(score)
            each_row.append(score2)
            whole_dataframe.append(each_row)
        whole_df = pd.DataFrame(whole_dataframe,columns = column_dataframe)
        whole_df.to_csv(self.subject_name+'_scores_sorted.csv',header=False,index=False)


def launch_tk_gui():
    # Create TK root object and GUI window.
    window = tk.Tk()
    Browser(window)
    window.mainloop()

if __name__ == "__main__":
    launch_tk_gui()