import pandas as pd
from sklearn import linear_model
from tkinter import *
from PIL import ImageTk,Image

train = pd.read_csv('Train.csv')
train_data = [train]

for dataset in train_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # summarize the name to get the title only

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
for dataset in train_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Title map: Mr : 0 , Miss : 1 , Mrs: 2 , Others: 3

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
    # fill missing age with median age for each title


sex_mapping = {"male": 0, "female": 1}
for dataset in train_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    # Sex map: Male : 0 , Female : 1

train.drop(['PassengerId',
            'Name',
            'SibSp',
            'Parch',
            'Ticket',
            'Cabin',
            'Embarked',
            'Title'],
           axis='columns',inplace=True)


x = train[['Pclass','Sex','Age','Fare']].astype(int)
y = train['Survived'].astype(int)

regr = linear_model.LinearRegression()
regr.fit(x,y)

#-----------------------------------------------------------------------------------------


root = Tk()
root.title("Survival Prediction")
canvas1 = Canvas(root,width=1000,height=630)
canvas1.pack()

img = Image.open('C:\\Users\\Win 10 Pro\\Documents\\Baba\\Data mining on titanic dataset\\myImage.jpg')
img_end = ImageTk.PhotoImage(img)
lbl = Label(root,image=img_end)
lbl.place(x=1,y=1)

frm1 = Frame(root,bg="Orange",bd=5)
frm1.place(relx=0.5,rely=0.1,relwidth=0.75,relheight=0.1,anchor='n')

lbl1 = Label(frm1,text="Choose passenger class :")
lbl1.place(relwidth=0.4,relheight=1)
lbl1.config(font=("Times",18))

a = IntVar()
rb1 = Radiobutton(frm1,text="1st",variable=a,value=1,font=("Times",12),bg="White")
rb1.place(relx=0.5, rely=0.2, relwidth=0.1, relheight=0.3)

rb2 = Radiobutton(frm1,text="2nd",variable=a,value=2,font=("Times",12),bg="White")
rb2.place(relx=0.7, rely=0.2, relwidth=0.1, relheight=0.3)

rb3 = Radiobutton(frm1,text="3nd",variable=a,value=3,font=("Times",12),bg="White")
rb3.place(relx=0.9, rely=0.2, relwidth=0.1, relheight=0.3)


frm2 = Frame(root,bg="Orange",bd=5)
frm2.place(relx=0.5,rely=0.25,relwidth=0.75,relheight=0.1,anchor='n')

lbl2 = Label(frm2,text="Choose passenger gender :")
lbl2.place(relwidth=0.4,relheight=1)
lbl2.config(font=("Times",18))

b = IntVar()
rb4 = Radiobutton(frm2,text="Male",variable=b,value=0,font=("Times",12),bg="White")
rb4.place(relx=0.56, rely=0.2, relwidth=0.1, relheight=0.3)

rb4 = Radiobutton(frm2,text="Female",variable=b,value=1,font=("Times",12),bg="White")
rb4.place(relx=0.8, rely=0.2, relwidth=0.1, relheight=0.3)


frm3 = Frame(root,bg="Orange",bd=5)
frm3.place(relx=0.5,rely=0.4,relwidth=0.75,relheight=0.1,anchor='n')

lbl3 = Label(frm3,text="Enter passenger age :")
lbl3.place(relwidth=0.4,relheight=1)
lbl3.config(font=("Times",18))

ent3 = Entry(frm3)
ent3.place(relx=0.5,relwidth=0.5,relheight=1)
ent3.config(font=("Times",18))


frm4 = Frame(root,bg="Orange",bd=5)
frm4.place(relx=0.5,rely=0.55,relwidth=0.75,relheight=0.1,anchor='n')

lbl4 = Label(frm4,text="Enter ticket fare :")
lbl4.place(relwidth=0.4,relheight=1)
lbl4.config(font=("Times",18))

ent4 = Entry(frm4)
ent4.place(relx=0.5,relwidth=0.5,relheight=1)
ent4.config(font=("Times",18))


frm5 = Frame(root,bg="Orange",bd=5)
frm5.place(relx=0.5,rely=0.75,relwidth=0.75,relheight=0.1,anchor='n')


def values():
    global newPclass
    newPclass = int(a.get())
    global newSex
    newSex = int(b.get())
    global newAge
    newAge = int(ent3.get())
    global newFare
    newFare = float(ent4.get())
    prediction_result = regr.predict([[newPclass,newSex,newAge,newFare]])
    if prediction_result <= 0.5:
        label_prediction = Label(frm5, text="Passenger didn't survive", bg="Black",fg="White")
        label_prediction.place(relx=0.5, relwidth=0.5, relheight=1)
        label_prediction.config(font=("Times", 18))
    else:
        label_prediction = Label(frm5, text="Passenger survived", bg="Black",fg="White")
        label_prediction.place(relx=0.5, relwidth=0.5, relheight=1)
        label_prediction.config(font=("Times", 18))


btn1 = Button(frm5,text="Pridect survival",command=values,bg="Black",fg="White")
btn1.place(relwidth=0.4,relheight=1)
btn1.config(font=("Times",18))


root.mainloop()