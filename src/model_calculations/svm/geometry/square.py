#this is a very specific square, both edges parallel to the axis and (0,0) one of the vertices and only non negative coordinates
from .line import Line

class Square:
    def __init__(self, length):
        self.length=length
        self.lines=[Line(0,1,0),Line(0,1,-length),Line(1,0,0),Line(1,0,-length)]

    def find_intersections(self,line):
        intersections=[]
        for line2 in self.lines:
            intersections.append(line.find_intersection(line2))
        constrained_intersections=[]
        for intersection in intersections:
            if intersection[0]<=self.length and intersection[0]>=0 and intersection[1]<=self.length and intersection[1]>=0:
                constrained_intersections.append(intersection)

        constrained_intersections.sort(key=lambda t: t[0])
        return (constrained_intersections[0],constrained_intersections[-1])