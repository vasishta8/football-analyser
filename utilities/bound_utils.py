def get_center(bound):
    x1,y1,x2,y2 = bound
    return int((x1+x2)//2), int((y1+y2)//2)

def get_width(bound):
    return bound[2] - bound[0]
