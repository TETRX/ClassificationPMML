from .a_optimizer import Optimizer
from .geometry.square import Square
from .geometry.line import Line
import random
import math

class SequentialOptimizer(Optimizer):

    def take_two(self, m):
        two=random.sample(range(m),2)
        return (two[0],two[1])

    def optimize(self, x,y,C,kernel_func):
        m=len(x)
        # alphas=[0.0 for i in range(m)]
        alphas=[]
        neg_counter=0
        pos_counter=0
        for y_i in y:
            if y_i>0:
                pos_counter+=1
            else:
                neg_counter+=1
        for y_i in y:
            if y_i>0:
                alphas.append((pos_counter/m)*C/m)
            else:
                alphas.append((neg_counter/m)*C/m)
        print(alphas)
        for _ in range(m):
            # print("---------")
            i,j=self.take_two(m)
            # print("ys",y[i],y[j])
            a=0
            b=0 #minimum of a quadratic function (ax^2+bx+c) is -b/2a
            zeta=alphas[i]*y[i]+alphas[j]*y[j]
            for l in range(m):
                if l==i or l==j:
                    continue #skip for now
                b+=-alphas[l]*y[l]*y[i]*kernel_func.compute(x[l],x[i])/2
                b+=alphas[l]*y[l]*y[j]*kernel_func.compute(x[l],x[j])*y[i]*y[j]/2
            # print("1",b)
            kappa=kernel_func.compute(x[i],x[j])
            a=-kappa
            b+=kappa*y[i]*zeta
            # print("2",b)

            b+=1+y[i]*y[j] # account for the first sum
            # print("3",b)
            # print("3",a)
            minimum=0
            if a!=0:
                minimum=-b/(2*a)
            else:
                minimum=math.inf*b
            # print(minimum)
            # L and H are two values of alpha_i such that (Ly_i,y_j(zeta-Ly_i)) lays on the [C/m,C/m] square, L<H
            square=Square(C/m)
            line=Line(y[j],y[i],-zeta)


            L_tuple,H_tuple=square.find_intersections(line)
            L,H=(L_tuple[0],H_tuple[0])
            constrained_minimum=0
            if minimum>H:
                constrained_minimum=H
            elif minimum<L:
                constrained_minimum=L
            else:
                constrained_minimum=minimum
            # print(constrained_minimum)
            alphas[i]=constrained_minimum
            alphas[j]=y[j]*(zeta-constrained_minimum*y[i])
            # print(alphas)
        print(alphas)
        return alphas
