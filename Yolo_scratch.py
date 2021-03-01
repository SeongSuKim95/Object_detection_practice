# Yolo Algorithm
# Each cell will output a prediction with a corresponding bounding box
# Select responsible cell --> Containing middle point of object 

# Each output and label will be relative to the cell
# Have : [x,y,w,h]  x,y -> Relative coordinate of midpoint (Between 0 and 1) ... w,h -> Can be greater than 1, if object is bigger than cell

# How the labels will actually look
# label_cell = [c1,c2,..., p, x, y, w, h ]
# p = Probability that there is an object, Note : A cell can only detect one object
# pred_cell = [c1,c2,...,p1, x, y, w, h, p2, x, y, w, h]

# Target shape for one image : (S, S, 25) (for 20 classes)
# Prediction shape for one image : (S, S, 30) (One additional)
