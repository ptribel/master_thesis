from turtle import *
from random import random, randint
t = Turtle()
t.hideturtle()
t.speed(0)
tracer(0,0)

Screen().bgcolor("black")

for c in ["white", "red", "gray"]:
    t.penup()
    t.goto(0, 0)#randint(-100, 100), randint(-100, 100))
    t.pendown()
    t.color(c)
    for i in range(100):
        way = 1# [-1, 1][randint(0, 1)]
        radius = randint(10, 30)
        for j in range(radius):
            t.forward(1)
            t.right(way*1)
        turn = randint(30, 90)
        t.left(-way*turn)
update()
input()
