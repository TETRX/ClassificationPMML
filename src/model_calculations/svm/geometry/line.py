class Line:
    def __init__(self,a,b,c): #y*a+x*b+c=0
        self.a=a
        self.b=b
        self.c=c

    def find_intersection(self, line): #does NOT handle edgecases
        intersection_x=0
        intersection_y=0
        if self.a==0:
            intersection_x=-self.c/self.b
            intersection_y=(-intersection_x*line.b-line.c)/line.a
        else:
            intersection_x=(self.c*line.a/self.a-line.c)/(line.b-self.b*line.a/self.a)
            intersection_y=(-intersection_x*self.b-self.c)/self.a
        return (intersection_x,intersection_y)