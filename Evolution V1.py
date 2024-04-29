from math import sqrt
import numpy
import pygame
import random
from pygame.locals import *
 
import os
 
from pygame.math import Vector2
from pygame.version import PygameVersion
 
main_dir = os.path.split(os.path.abspath(__file__))[0]
 
pygame.init()
 
font20 = pygame.font.Font("freesansbold.ttf", 20)
font8 = pygame.font.Font("freesansbold.ttf", 8)
 
FASTFPS = 10000
SLOWFPS = 15
 
WIDTH, HEIGHT = 800,800
CENTREWIDTH, CENTREHEIGHT = WIDTH // 2, HEIGHT // 2
 
SIDEBARWIDTH = 300
INFOWIDTH,INFOHEIGHT = SIDEBARWIDTH, HEIGHT * 0.2
GRAPHWIDTH,GRAPHHEIGHT = SIDEBARWIDTH, HEIGHT * 0.3
NNDIAGRAMWIDTH,NNDIAGRAMHEIGHT = SIDEBARWIDTH, HEIGHT * 0.4
 
MAXTIME = 6000
CREATURETIMEOUT = 12
CREATUREENERGYCOST = 0.05
 
USEACCEL = True
 
STARTINGSIZE = 10
SPAWNDISTANCE = HEIGHT * 0.01
SIMULATIONCOUNT = 2
 
LINEGRAPHRATE = 250
LINEGRAPHMAX = GRAPHWIDTH
LINEGRAPHRESET = False
 
screen = pygame.display.set_mode((WIDTH + SIDEBARWIDTH,HEIGHT), HWSURFACE | DOUBLEBUF)
lineGraphSurface = pygame.surface.Surface((GRAPHWIDTH, GRAPHHEIGHT))
nnDiagramSurface = pygame.surface.Surface((NNDIAGRAMWIDTH, NNDIAGRAMHEIGHT))
infoSurface = pygame.surface.Surface((INFOWIDTH, INFOHEIGHT))
sidebarSurface = pygame.surface.Surface((SIDEBARWIDTH, HEIGHT))
 
pygame.display.set_caption("Evolution")
 
clock = pygame.time.Clock()
 
BLACK = (0,0,0)
DARKGREY = (20,20,20)
MIDGREY = (77,77,77)
LIGHTGREY = (144,144,144)
WHITE = (255,255,255)
 
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,0,255)
ORANGE = (255,144,0)
PURPLE = (144,0,255)
CYAN = (0,255,255)
MAGENTA = (255,0,255)
YELLOW = (255,255,0)
LIGHTBLUE = (20,20,255)
 
ListOfColors = [RED, GREEN, BLUE, ORANGE, PURPLE, MAGENTA, YELLOW, WHITE]
 
CreaturesList = [[] for i in range(SIMULATIONCOUNT)]
lineGraphsList = []
 
def linear(x):
    return x
 
def relu(x):
    return numpy.maximum(0, x)
 
def sigmoid(x):
    return 1 / (1 + (2.71828 ** (-x)))
 
def tanh(x):
    return numpy.tanh(x)
 
def log(x):
    if x > 0:
        return numpy.log(x)
    else: 
        return 0
 
def sin(x):
    return numpy.sin(x)
 
def gaussian(x):
    return 2.71828 ** ((x ** 2) * -1)
 
def abs(x):
    return numpy.abs(x)
 
def Latch(x):
    if x >= 1:
        return 1
    else:
        return 0
 
ActivationFunctionDict = {
    0 : linear,
    1 : relu,
    2 : sigmoid,
    3 : tanh,
    4 : log,
    5 : sin,
    6 : gaussian,
    7 : abs,
    8 : Latch,
}
 
def MaxAbs(x,y):
    if abs(x) > abs(y):
        return x
    else:
        return y
 
def MinAbs(x,y):
    if abs(x) < abs(y):
        return x
    else:
        return y
 
def loadImage(file):
    file = os.path.join(main_dir, "Images", file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit(f'Could not load image "{file}" {pygame.get_error()}')
    return surface.convert()
 
NNInputNames = {
    0: "Cons",
    1: "Angl",
    2: "Vel",
    3: "Time",
    4: "XPos",
    5: "YPos",
}
 
NNNames = {
    0: "Lin",
    1: "Relu",
    2: "Sig",
    3: "Tanh",
    4: "Log",
    5: "Sin",
    6: "Gaus",
    7: "Abs",
    8: "Ltch",
}
NNOutputNames = {
    0: "Acc",
    1: "Rot",
    2: "Kill",
}
 
#=======================================================================================================================================
 
class layer:
    def __init__(self, n_inputs, n_nodes):
        #n_inputs is the number of nodes in the previous layer, and n_nodes is the number of nodes in this current layer
 
        #weights array is a 2d array that multiplies the activation value from each node in the previous layer, for each node
        #biasses array contains each node's bias (flat value added to activation value) for each node
        #node array contains each of the node's value
        self.weightsArray = [[0 for i in range(n_inputs)] for j in range(n_nodes)]
 
        self.activationFunctionArray = [1 for i in range(n_nodes)]
        self.biasesArray = [0 for i in range(n_nodes)]
        self.nodeArray = [0 for i in range(n_nodes)]
 
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
 
    
    def ForwardPass(self, inputsArray):
        #overwrites nodearray
        self.nodeArray = [0 for i in range(self.n_nodes)]
 
        for i in range(self.n_nodes):
            #sum values of weights * inputs
            for j in range(self.n_inputs):
                #print("g1 NNodes {} . NInputs {} NodeArray {} . InputArray {} . weightsArray {} . i{} . j{}".format(self.n_nodes, self.n_inputs, self.nodeArray, inputsArray, self.weightsArray, i, j))
                self.nodeArray[i] += self.weightsArray[i][j] * inputsArray[j]
 
            #add the bias in biasesarray to the sum
            self.nodeArray[i] += self.biasesArray[i]
 
    def InheritLayer(self, ParentLayer):
        self.weightsArray = [x[:] for x in ParentLayer.weightsArray] #copy weights
        self.biasesArray = ParentLayer.biasesArray[:] #copy biases
        self.activationFunctionArray = ParentLayer.activationFunctionArray[:] #copy activation functions
 
    def MutateLayer(self, mutationStrength, mutationChance):
        for i in range(self.n_nodes):
            for j in range(self.n_inputs):
 
                if random.randint(1,1000) <= mutationChance: #mutate weights
                    if random.randint(1, 5) == 1: # 20% chance to flip weight
                        self.weightsArray[i][j] = self.weightsArray[i][j] * -1
                    else:
                        self.weightsArray[i][j] += (random.randint(-100, 100) / 100) * mutationStrength #change weight
 
            if random.randint(1,1000) <= mutationChance:
                if random.randint(1, 5) == 1: # 20% chance to flip bias
                        self.biasesArray[i] = self.biasesArray[i] * -1
                else:
                    self.biasesArray[i] += (random.randint(-100, 100) / 100) * mutationStrength #change bias
 
            if random.randint(1,1000) <= mutationChance:
                self.activationFunctionArray[i] = random.randint(0,8) #change activation function
 
    def Activation(self):
        for i in range(self.n_nodes):
            self.nodeArray[i] = ActivationFunctionDict[self.activationFunctionArray[i]](self.nodeArray[i])
            
 
#=======================================================================================================================================
class Creature:
    def __init__(self, posx, posy, width, height, speed, maxEnergy, color):
        #Mutatable stats
        self.species = ""
        self.Generation = 0
        self.MaxHealth = 100
        self.Health = self.MaxHealth
        self.MaxSpeed = speed
        self.MaxEnergy = maxEnergy
        self.AttackRange = 10
        self.color = color
        self.outlineColor = (numpy.clip(self.color[0] - 60, 0, 255), numpy.clip(self.color[1] - 60, 0, 255), numpy.clip(self.color[2] - 60, 0, 255))
 
        #Body
        self.width = width
        self.height = height
        self.BodyRect = pygame.Rect(posx, posy, width, height)
 
        self.TimeAlive = 0
        self.Energy = maxEnergy/3
 
        #position & physics
        self.pos = pygame.Vector2(self.BodyRect.center)
        self.velocity = 0
        self.angle = 40
        self.direction = pygame.Vector2(1, 0)
 
        #Neural Network
        self.networkShape = [6,6,6,6,3]
 
        self.layers = []
        for i in range(len(self.networkShape) - 1):
            #ignoring the first layer(input layer), make the nodes for each layer and make the weights and biases to the new nodes
            #using the previous layer
            self.layers.append(layer(self.networkShape[i], self.networkShape[i + 1]))
 
    #Neural Net Brain calculates every value for the neural net
    def NNBrain(self,inputs):
        for i in range(len(self.layers)):
            if i == 0:
                #print("layer {}, fpass input {}, output {}".format(self.layers[i], inputs, self.layers[i].ForwardPass(inputs[i])) )
                self.layers[i].ForwardPass(inputs)
                self.layers[i].Activation()
                #print("Inputs done, {}". format(inputs))
                #print("Hidden done, {}". format(self.layers[i].nodeArray))
 
            elif i == len(self.layers) - 1:
                self.layers[i].ForwardPass(self.layers[i - 1].nodeArray)
                #print("Outputs done, {}". format(self.layers[i].nodeArray))
                #print("==============================================================")
 
            else:
                self.layers[i].ForwardPass(self.layers[i - 1].nodeArray)
                self.layers[i].Activation()
                #print("Hidden done, {}". format(self.layers[i].nodeArray))
 
    def proximity(self, CreatureslistIndex):
        for C in CreaturesList[CreatureslistIndex]:
            if pygame.Vector2.distance_to(self.pos, C.pos) <= self.AttackRange:
                return 1
            else:
                return 0
 
    def Mutate(self, mutationStrength, mutationChance):
        for i in range(len(self.layers)):
            self.layers[i].MutateLayer(mutationStrength, mutationChance)
        if random.randint(1,1000) <= mutationChance:
            self.MaxSpeed += random.randint(-100, 100) / 100
 
        if random.randint(1,1000) <= mutationChance:
            self.MaxEnergy += random.randint(-100,100)
 
        if random.randint(1,100) <= 2:
            self.color = (numpy.clip(self.color[0] + random.randint(-60,60), 0, 255),numpy.clip(self.color[1] + random.randint(-60,60), 0, 255),numpy.clip(self.color[2] + random.randint(-60,60), 0, 255))
            self.UpdateColor()
 
        
    def InheritNetwork(self, ParentCreature):
        for i in range(len(self.layers)):
            self.layers[i].InheritLayer(ParentCreature.layers[i])
 
    def UpdateColor(self):
        self.outlineColor = (numpy.clip(self.color[0] - 60, 0, 255), numpy.clip(self.color[1] - 60, 0, 255), numpy.clip(self.color[2] - 60, 0, 255))
 
    #Moves the creature forward by velocity
    def Accelerate(self, acceleration):
        maxAccel = MinAbs(acceleration, self.MaxSpeed / 100) #gets the value closest to 0, ensuring that the acceleration does not increase speed past MaxSpeed
        if self.velocity + maxAccel < self.MaxSpeed: #checks if the new velocity will be above MaxSpeed
            self.velocity += maxAccel #adds velocity
        else:
            self.velocity = self.MaxSpeed #sets the velocity to MaxSpeed, capping the velocty
 
    def SetVelocity(self, inputValue):
        if inputValue < self.MaxSpeed:
            self.velocity = inputValue
        else:
            self.velocity = self.MaxSpeed
 
    #moves the creature in the direction it is facing
    def MoveSelf(self): 
        direction = pygame.Vector2(0, self.velocity).rotate(-self.angle)
        self.pos += direction
        self.Energy -= (abs(self.velocity) / self.MaxSpeed) - 0.2
 
    #rotates the creature
    def RotateSelf(self, AngleToRotate):
        if abs(self.velocity) > 0: 
            self.angle += AngleToRotate * (3/numpy.minimum(self.velocity,3)) #makes creatures moving faster rotate slower
        else:
            self.angle += AngleToRotate * 10 #necassary to avoid divide by 0 errors
        
        if self.angle > 180: #keeps the angle between -180 and 180
            self.angle = round(-180 + (self.angle % 180), 3)
        elif self.angle < -180:
            self.angle = round(180 - (numpy.abs(self.angle) % 180), 3)
 
    def kill(self, CreatureslistIndex, CurrentlyViewedSimulation):
        centrePos = Vector2(self.pos.x + (self.width / 2), self.pos.y + (self.height / 2))
        for C in CreaturesList[CreatureslistIndex]:
            if C != self:
                if pygame.Vector2.distance_to(centrePos, Vector2(C.pos.x + (C.width / 2),C.pos.y + (C.height / 2))) <= self.AttackRange and C.TimeAlive >= 100:
                    if C.Health > 0:
                        C.Health -= 30
                        self.Energy += C.Energy / 50
 
                        if CreatureslistIndex == CurrentlyViewedSimulation - 1 or CurrentlyViewedSimulation == 0:
                            pygame.draw.circle(screen,YELLOW,centrePos, self.AttackRange, 1)
                    else:
                        self.Energy += C.Energy * 0.8
                        CreaturesList[CreatureslistIndex].remove(C)
                        if CreatureslistIndex == CurrentlyViewedSimulation - 1 or CurrentlyViewedSimulation == 0:
                            pygame.draw.circle(screen,YELLOW, centrePos, self.AttackRange, 1)
                else:
                    if CreatureslistIndex == CurrentlyViewedSimulation - 1 or CurrentlyViewedSimulation == 0:
                        pygame.draw.circle(screen,RED, centrePos, self.AttackRange, 1)
 
        self.Energy -= self.MaxEnergy / 100
 
    def display(self):
        pygame.draw.rect(screen, self.color, self.BodyRect)
    
    def displayMovement(self):
        centrePosx = self.pos.x + (self.width / 2)
        centrePosy = self.pos.y + (self.height / 2)
        pygame.draw.aaline(screen, self.outlineColor, Vector2(centrePosx, centrePosy), Vector2(centrePosx + ((self.velocity * 15) * numpy.cos(self.angle * -0.0174533 + 1.570797)), centrePosy + ((self.velocity * 15) * numpy.sin(self.angle * -0.0174533 + 1.570797))), 2)
 
    def update(self, CreatureslistIndex, CurrentlyViewedSimulation):
        
        #self.Accelerate(self.layers[len(self.layers) - 1].nodeArray[0])
        #self.SetVelocity(self.layers[len(self.layers) - 1].nodeArray[0])
        self.velocity = self.layers[len(self.layers) - 1].nodeArray[0]
        self.MoveSelf()
        self.RotateSelf(self.layers[len(self.layers) - 1].nodeArray[1])
        #self.angle = (self.layers[len(self.layers) - 1].nodeArray[1] * 180)
        if self.layers[len(self.layers) - 1].nodeArray[2] >= 1: 
            self.kill(CreatureslistIndex, CurrentlyViewedSimulation)
 
        self.TimeAlive += 1
 
        self.BodyRect = (self.pos.x, self.pos.y, self.width, self.height)
 
    def getRect(self):
        return self.BodyRect
 
#=======================================================================================================================================    
class LineGraph:
    def __init__(self, posx, posy, width, height, color, lineWeight, data):
        self.pos = pygame.Vector2(posx, posy)
        self.width = width
        self.height = height
        self.color = color
        self.lineWeight = lineWeight
        self.data = data
        self.lineGraphPoints = 1
 
    def meanList(self, inputData):
        pass
 
    def getScale(self, inputData):
        if len(inputData) >= 2:
            SortedPointsx = sorted(inputData, key = lambda x: x[0], reverse = False)
            SortedPointsy = sorted(inputData, key = lambda x: x[1], reverse = False)
 
            minY = SortedPointsy[0][1]
            maxY = SortedPointsy[len(SortedPointsy) - 1][1]
            
            xScale = self.width / SortedPointsx[len(SortedPointsx) - 1][0]
            if maxY - minY != 0:
                yScale = self.height / (maxY - minY)
                return (xScale,yScale,minY)
            else:
                return(xScale,1,minY)
 
    def draw(self, inputData, scale):
        if len(inputData) >= 2:
            ScaledPoints = []
            SortedPointsx = sorted(inputData, key = lambda x: x[0], reverse = False)
 
            for i in SortedPointsx:
                scaledX = (i[0] * scale[0]) + self.pos.x #multiplies the scale of the x points to fit the 
                scaledY = ((-i[1] + scale[2]) * scale[1]) + self.pos.y #same as the previous line, but also multiplies by -1 to make the lineGraph the right way up, and adds an offset to keep it within the lineGraph's bounds
                ScaledPoints.append([scaledX, scaledY])
 
            pygame.draw.lines(lineGraphSurface, self.color, False, ScaledPoints, self.lineWeight)
 
#======================================================================================================================================= 
 
class NNDiagram:
    def __init__(self, posx, posy, width, height, nodeSize, layers, networkShape):
        self.pos = pygame.Vector2(posx, posy)
        self.width = width
        self.height = height
        self.nodeSize = nodeSize
        self.layers = layers
        self.networkShape = networkShape
 
    def NodePoint(self, layernum, layermax, nodenum, nodemax): #gets the point to place a node
        nodex = self.pos.x + (layernum * (self.width / layermax))
        nodey = self.pos.y + (nodenum * (self.height / nodemax))
        return Vector2(nodex,nodey)
 
    def NodeColor(self, inputvalue, colorMult): #gets the colour of a node or weight, green for positive, red for negative
        outputColor = (122,122,122)
        if inputvalue >= 0:
            redvalue = numpy.clip(outputColor[0] - (inputvalue * colorMult), 0, 225)
            greenvalue = numpy.clip(outputColor[1] + (inputvalue * colorMult), 0, 225)
            bluevalue = numpy.clip(outputColor[2] - (inputvalue * colorMult), 0, 225)
            outputColor = (redvalue,greenvalue,bluevalue)
        else:
            redvalue = numpy.clip(outputColor[0] - (inputvalue * colorMult), 0, 225)
            greenvalue = numpy.clip(outputColor[1] + (inputvalue * colorMult), 0, 225)
            bluevalue = numpy.clip(outputColor[2] + (inputvalue * colorMult), 0, 225)
            outputColor = (redvalue,greenvalue,bluevalue)
 
        return(outputColor)
 
    def drawWeights(self): #draw weights, colour determined by multiplier
        for i in range(len(self.networkShape)):
            if i != 0:
                for j in range(self.networkShape[i]):
                    thisNodePos = self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1)
                    for k in range(self.networkShape[i - 1]):
                        prevNodePos = self.NodePoint(i - 1, len(self.networkShape) - 1, k, self.networkShape[i - 1] - 1)
                        thisWeightMult = self.layers[i - 1].weightsArray[j][k]
 
                        if abs(thisWeightMult) > 0.3:
                            pygame.draw.line(nnDiagramSurface, self.NodeColor(thisWeightMult,50), prevNodePos, thisNodePos, 2)
 
    def drawWeightsColored(self, inputs): #draw weights, colour determined by current value passing through it
        for i in range(len(self.networkShape)):
            if i == 1:
                for j in range(self.networkShape[i]):
                    thisNodePos = self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1)
                    for k in range(self.networkShape[0]):
                        prevNodePos = self.NodePoint(0, len(self.networkShape) - 1, k, self.networkShape[0] - 1)
                        thisWeightMult = 0
                        thisWeightMult = self.layers[0].weightsArray[j][k] * inputs[k]
 
                        if abs(thisWeightMult) > 0.2:
                            pygame.draw.line(nnDiagramSurface, self.NodeColor(thisWeightMult,50), prevNodePos, thisNodePos, 2)
 
            elif i >= 2:
                for j in range(self.networkShape[i]):
                    thisNodePos = self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1)
                    for k in range(self.networkShape[i - 1]):
                        prevNodePos = self.NodePoint(i - 1, len(self.networkShape) - 1, k, self.networkShape[i - 1] - 1)
                        thisWeightMult = self.layers[i - 1].weightsArray[j][k] * self.layers[i - 2].nodeArray[k]
                        if abs(thisWeightMult) > 0.2:
                            pygame.draw.line(nnDiagramSurface, self.NodeColor(thisWeightMult,50), prevNodePos, thisNodePos, 2)
 
    def DrawNodes(self, inputs, InputColorMult, HiddenColorMult, OutputColorMult, showDetail): #draw nodes, colour determined by current value passing through it
        for i in range(len(self.networkShape)):
            if i == 0:
                for j in range(self.networkShape[i]):
                    thisNodePos = Vector2(self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1))
                    pygame.draw.circle(nnDiagramSurface, self.NodeColor(inputs[j], InputColorMult), (thisNodePos.x,thisNodePos.y), self.nodeSize)
                    if showDetail == True:
                        DrawTextSmall(nnDiagramSurface,NNInputNames[j], LIGHTBLUE, (thisNodePos.x - 8,thisNodePos.y - 8))
                        DrawTextSmall(nnDiagramSurface, str(round(inputs[j],2)), BLACK, (thisNodePos.x - 8,thisNodePos.y))
                    else:
                        DrawTextSmall(nnDiagramSurface,NNInputNames[j], BLACK, (thisNodePos.x - 8,thisNodePos.y - 4))
    
            elif i == len(self.networkShape) - 1:
                for j in range(self.networkShape[i]):
                    thisNodePos = Vector2(self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1))
                    pygame.draw.circle(nnDiagramSurface, self.NodeColor(self.layers[i - 1].nodeArray[j], OutputColorMult), (thisNodePos.x,thisNodePos.y), self.nodeSize)
                    if showDetail == True:
                        DrawTextSmall(nnDiagramSurface,NNOutputNames[j], LIGHTBLUE, (thisNodePos.x - 8,thisNodePos.y - 8))
                        DrawTextSmall(nnDiagramSurface, str(round(self.layers[i - 1].nodeArray[j],2)), BLACK, (thisNodePos.x - 8,thisNodePos.y ))
                    else:
                        DrawTextSmall(nnDiagramSurface,NNOutputNames[j], BLACK, (thisNodePos.x - 8,thisNodePos.y - 4))
                    
            else:
                for j in range(self.networkShape[i]):
                    thisNodePos = Vector2(self.NodePoint(i, len(self.networkShape) - 1, j, self.networkShape[i] - 1))
                    pygame.draw.circle(nnDiagramSurface, self.NodeColor(self.layers[i - 1].nodeArray[j], HiddenColorMult), (thisNodePos.x,thisNodePos.y), self.nodeSize)
                    if showDetail == True:
                        DrawTextSmall(nnDiagramSurface,NNNames[self.layers[i - 1].activationFunctionArray[j]], LIGHTBLUE, (thisNodePos.x - 8,thisNodePos.y - 8))
                        DrawTextSmall(nnDiagramSurface,str(round(self.layers[i - 1].nodeArray[j],2)), BLACK, (thisNodePos.x - 8,thisNodePos.y ))
                    else:
                        DrawTextSmall(nnDiagramSurface,NNNames[self.layers[i - 1].activationFunctionArray[j]], BLACK, (thisNodePos.x - 8,thisNodePos.y - 4))
                    
 
#======================================================================================================================================= 
 
class CreatureInfo:
    def __init__(self, posx,posy, width, height, selectedCreature):
        self.pos = pygame.Vector2(posx, posy)
        self.width = width
        self.height = height
        self.selectedCreature = selectedCreature
 
    def Update(self): #draw a panel with info on the current creature
        creatureColour = self.selectedCreature.color
        pygame.draw.rect(infoSurface, creatureColour, (2, 2, 25, 25))
        pygame.draw.rect(infoSurface, self.selectedCreature.outlineColor, (2, 2, 25, 25), 3)
        DrawText(infoSurface, f"Generation: {self.selectedCreature.Generation}", PURPLE, (0,30))
        DrawText(infoSurface, f"Time: {self.selectedCreature.TimeAlive} / {MAXTIME}", CYAN, (0,50))
        DrawText(infoSurface, f"Health: {round(self.selectedCreature.Health, 1)} / {self.selectedCreature.MaxHealth}", RED, (0,70))
        DrawText(infoSurface, f"Speed: {round(self.selectedCreature.velocity, 1)} / {round(self.selectedCreature.MaxSpeed, 1)}", ORANGE, (0,90))
        DrawText(infoSurface, f"Energy: {round(self.selectedCreature.Energy, 1)} / {self.selectedCreature.MaxEnergy}", YELLOW, (0,110))
 
#======================================================================================================================================= 
 
class Button: 
    def __init__(self, color, posx, posy, width, height, text, pressedFunction):
        self.rect = pygame.Rect(posx, posy, width, height)
        self.color = color
        self.hoverColor = numpy.clip(self.color[0] - 20, 0, 255), numpy.clip(self.color[1] - 20, 0, 255), numpy.clip(self.color[2] - 20, 0, 255)
        self.pressedColor = numpy.clip(self.color[0] - 60, 0, 255), numpy.clip(self.color[1] - 60, 0, 255), numpy.clip(self.color[2] - 60, 0, 255)
        self.width = width
        self.height = height
        self.text = text
        self.pressedfunction = pressedFunction
 
    def Hover():
        pass
 
#======================================================================================================================================= 
def Instructions():
    print("")
    print("============================================ HOTKEYS ============================================")
    print("Left click creature: Select a creature to be viewed")
    print("Space: Pause")
    print("Tab: Cycle viewed simulation")
    print("Enter: View all simulations")
    print("Backspace: Claer currently viewed simulation(s)")
    print("C: Print number of creatures in viewed simulation(s)")
    print("G: Print highest generation in currently viewed simulation(s)")
    print("R: Clear currently viewed graph(s)")
    print("P: Toggle graph visibility (Slightly improves performance)")
    print("O: Toggle brain diagram visibility (SIGNIFICANTLY improves performance)")
    print("I: Toggle creature Stats (Slightly improves performance)")
    print("U: Toggle creature Movement Indicator (Slightly improves performance)")
    print("D: Toggle detailed brain diagrams (Slightly improves performance)")
    print("S: Toggle slow mode")
    print("Escape: Quit program")
    print("=================================================================================================")
    print("")
    
 
def DrawText(surface, txt, color, pos): 
    text = font20.render(txt, 1, pygame.Color(color))
    pygame.Surface.blit(surface, text, pos)
 
def DrawTextSmall(surface, txt, color, pos):
    text = font8.render(txt, 1, pygame.Color(color))
    pygame.Surface.blit(surface, text, pos)
 
def DrawLineGraphs(LineGraphToShow):
    if len(lineGraphsList[0].data) >= 2:
        lineGraphSurface.fill((0,0,0,0))
 
        combinedData = []
        for G in lineGraphsList:
            for i in G.data:
                combinedData.append(i)
        
        SortedCombinedData = sorted(combinedData, key = lambda x: x[1], reverse = True)
 
        if len(SortedCombinedData) >= 2:
            combinedScale = lineGraphsList[0].getScale(SortedCombinedData)
 
            if LineGraphToShow == 0:
                for G in lineGraphsList:
                    G.draw(G.data, combinedScale)
            else:
                lineGraphsList[LineGraphToShow - 1].draw(lineGraphsList[LineGraphToShow - 1].data, combinedScale)
 
def DrawSidebar():
    pygame.Surface.fill(sidebarSurface,MIDGREY)
 
def RenderLineGraphs():
    pygame.Surface.blit(screen, lineGraphSurface,(WIDTH,HEIGHT - GRAPHHEIGHT))
 
def RenderSidebar():
    pygame.Surface.blit(screen, sidebarSurface,(WIDTH,0))
 
def RenderNNDiagram():
    pygame.Surface.blit(screen, nnDiagramSurface,(WIDTH, HEIGHT - GRAPHHEIGHT - NNDIAGRAMHEIGHT))
 
def RenderInfo():
    pygame.Surface.blit(screen, infoSurface,(WIDTH, HEIGHT - GRAPHHEIGHT - NNDIAGRAMHEIGHT - INFOHEIGHT))
 
class Menu:
    def __init__(self):
        self.Buttons = []
 
    def MenuInputs(self, mousePos): 
        for event in pygame.event.get():
            if event.type == pygame. QUIT:
                pygame.quit()
 
            if event.type == pygame.MOUSEBUTTONUP:
                for B in self.Buttons:
                    if pygame.Rect.collidepoint(B.rect, mousePos.x, mousePos.y):
                        B.pressedFunction()
 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
 
    def Update(self):
        mousepos = pygame.mouse.get_pos()
        self.MenuInputs(mousepos)
    
 
class Game:
    def __init__(self):
        self.fps = FASTFPS
        self.paused = False
        self.showLineGraphs = True
        self.showNNDiagram = True
        self.showDetail = False
        self.showInfo = True
        self.showMovement = True
 
        self.TotalTime = 0
 
        self.CurrentlyViewedSimulation = 0
        self.selectedCreature = None
 
        self.cInfo = CreatureInfo(0,0,INFOWIDTH,INFOHEIGHT,None)
        self.nnd = NNDiagram(20,20,NNDIAGRAMWIDTH - 40,NNDIAGRAMHEIGHT - 40, 10, None, [])
 
        self.RewardMax = 2
        self.RewardFunction = self. FarRewardFunction #------------REWARD FUNCTION---------------
        self.RewardVisualizerFunction = self.FarRewardDistance
        self.RewardVisualizers = []
 
    def GetRewardDistance(self, m):
        self.RewardVisualizers.clear()
        for i in range(7):
            self.RewardVisualizers.append(self.RewardVisualizerFunction(self.RewardMax / (2 ** i) + (len(CreaturesList[m]) * CREATUREENERGYCOST) , m))
 
    def CloseRewardFunction(self, x):
        return numpy.minimum(60 / x, self.RewardMax)
 
    def CloseRewardDistance(self, y, m):
        return 60 / y 
 
    def FarRewardFunction(self, x):
        return numpy.minimum((x ** 2) / 130000, self.RewardMax)
 
    def FarRewardDistance(self, y, m):
        return sqrt(130000 * y)
 
    def SetSelectedCreature(self, creatureToSelect):
        self.selectedCreature = creatureToSelect
        self.nnd.layers = creatureToSelect.layers
        self.nnd.networkShape = creatureToSelect.networkShape
        self.cInfo.selectedCreature = creatureToSelect
 
    def SimInputs(self): #processes inputs while the simulation is running
        for event in pygame.event.get():
                if event.type == pygame. QUIT:
                    pygame.quit()
 
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mousepos = pygame.mouse.get_pos()
                    for m in range(len(CreaturesList)):
                        for C in CreaturesList[m]:
                            if mousepos[0] > C.pos.x and mousepos[0] < C.pos.x + C.width and mousepos[1] > C.pos.y and mousepos[1] < C.pos.y + C.height:
                                if m == self.CurrentlyViewedSimulation - 1 or self.CurrentlyViewedSimulation == 0:
                                    self.SetSelectedCreature(C)
 
                                    print("CREATURE CLICKED")
 
    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        if self.CurrentlyViewedSimulation < len(CreaturesList):
                            self.CurrentlyViewedSimulation += 1
                        else:
                            self.CurrentlyViewedSimulation = 0
                        
                        DrawLineGraphs(self.CurrentlyViewedSimulation)
                        if self.showLineGraphs:
                            RenderLineGraphs()
    
                    elif event.key == pygame.K_RETURN:
                        self.CurrentlyViewedSimulation = 0
 
                    elif event.key == pygame.K_SPACE:
                        if self.paused == True:
                            self.paused = False
                        else:
                            self.paused = True
    
                    elif event.key == pygame.K_BACKSPACE:
                        if self.CurrentlyViewedSimulation == 0:
                            for i in CreaturesList:
                                    i.clear()
                                    TotalTime = 0
                            for G in lineGraphsList:
                                G.data = []
                                G.lineGraphPoints = 0
                        else:
                            CreaturesList[self.CurrentlyViewedSimulation - 1].clear()
                        print("Simulation restarted")    
    
                    elif event.key == pygame.K_c:
                        if self.CurrentlyViewedSimulation == 0:
                            totalC = 0
                            for i in CreaturesList:
                                totalC += len(i)
                            print("Total Creature count: {}".format(totalC))
    
                        else:
                            print("Creature count: {}".format(len(CreaturesList[self.CurrentlyViewedSimulation - 1])))
    
                    elif event.key == pygame.K_g:
                        HighestGenerationCreaturesList = []
                        for m in CreaturesList:
                            HighestGenerationCreaturesList.append(sorted(m, key = lambda x: x.Generation, reverse = True)[0])
                        HighestGenerationCreature = sorted(HighestGenerationCreaturesList, key = lambda x: x.Generation, reverse = True)[0]
    
                        self.SetSelectedCreature(HighestGenerationCreature)
                        
                        print("Highest generation: {}".format(HighestGenerationCreature.Generation))
    
                    elif event.key == pygame.K_r:
                        for G in lineGraphsList:
                            G.data = []
                            G.lineGraphPoints = 0
    
                    elif event.key == pygame.K_p:
                        if self.showLineGraphs == True:
                            self.showLineGraphs = False
                            lineGraphSurface.fill((0,0,0,0))
                        else:
                            self.showLineGraphs = True
 
                    elif event.key == pygame.K_o:
                        if self.showNNDiagram == True:
                            self.showNNDiagram = False
                        else:
                            self.showNNDiagram = True
 
                    elif event.key == pygame.K_i:
                        if self.showInfo == True:
                            self.showInfo = False
                            self.selectedCreature = None
                        else:
                            self.showInfo = True
 
                    elif event.key == pygame.K_u:
                        if self.showMovement == True:
                            self.showMovement = False
                        else:
                            self.showMovement = True
 
                    elif event.key == pygame.K_s:
                        if self.fps == FASTFPS:
                            self.fps = SLOWFPS
                        else:
                            self.fps = FASTFPS
 
                    elif event.key == pygame.K_d:
                        if self.showDetail == True:
                            self.showDetail = False
                        else:
                            self.showDetail = True
    
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()              
 
    def Update(self):
        Instructions()
        running = True
        self.paused = False
        pygame.key.set_repeat(200,200)
    
        for i in range(SIMULATIONCOUNT):
            lineGraphsList.append(LineGraph(0, GRAPHHEIGHT, GRAPHWIDTH, GRAPHHEIGHT, ListOfColors[i % len(ListOfColors)],1,[[0,0]]))
    
        while running:
 
            self.SimInputs()
 
            if self.paused == False:
                screen.fill(DARKGREY)
 
                for i in range(len(self.RewardVisualizers)):
                    pygame.draw.circle(screen, MIDGREY, (CENTREWIDTH, CENTREHEIGHT), self.RewardVisualizers[i], 1)
                    DrawTextSmall(screen, str(self.RewardMax / (2 ** i)), MIDGREY, (CENTREWIDTH, CENTREHEIGHT + self.RewardVisualizers[i]))
 
                if self.TotalTime % LINEGRAPHRATE == 0:
                    if self.CurrentlyViewedSimulation == 0:
                        self.GetRewardDistance(0)
                    else:
                        self.GetRewardDistance(self.CurrentlyViewedSimulation - 1)
                    
 
                    for i in range(len(lineGraphsList)):
                        lineGraphsList[i].data.append([lineGraphsList[i].lineGraphPoints, len(CreaturesList[i])])
                        lineGraphsList[i].lineGraphPoints += 1
                        if len(lineGraphsList[i].data) >= LINEGRAPHMAX:
                            for j in range(int(LINEGRAPHMAX // 2)):
                                #avg = (lineGraphsList[0].data[i][1] + lineGraphsList[0].data[i + 1][1]) / 2
                                #lineGraphsList[0].data[i][1] = avg
                                del lineGraphsList[i].data[j]
                            
                    DrawLineGraphs(self.CurrentlyViewedSimulation)
                
                for m in range(len(CreaturesList)):
                    if len(CreaturesList[m]) <= 0:
                        if LINEGRAPHRESET == True:
                            for G in lineGraphsList:
                                G.data = []
                                G.lineGraphPoints = 0
                        for i in range(STARTINGSIZE):
                            CreaturesList[m].append(Creature(random.randint(CENTREWIDTH - SPAWNDISTANCE, CENTREWIDTH + SPAWNDISTANCE),random.randint(CENTREHEIGHT - SPAWNDISTANCE, CENTREHEIGHT + SPAWNDISTANCE), 10, 10, 10, 300, (random.randint(0,255),random.randint(0,255),random.randint(0,255))))
                            CreaturesList[m][i].Mutate(3,900)
                            CreaturesList[m][i].angle = random.randint(-180,180)
                    
                    for C in CreaturesList[m]:
                        C.NNBrain([1, C.angle / 180, C.velocity / C.MaxSpeed, C.TimeAlive / MAXTIME, (C.pos.x - CENTREWIDTH) / CENTREWIDTH, (C.pos.y - CENTREHEIGHT) / CENTREHEIGHT])
                        #redcolor = numpy.clip(C.layers[len(C.layers) - 1].nodeArray[2] * 250, 0, 255)
                        #bluecolor = numpy.clip((abs(C.layers[len(C.layers) - 1].nodeArray[1] * 10)) - redcolor,0, 255)
                        #greencolor = numpy.clip((abs(C.layers[len(C.layers) - 1].nodeArray[0] * 10)) - redcolor, 0, 255)
                        #C.color = (redcolor, greencolor, bluecolor)
                        C.update(m, self.CurrentlyViewedSimulation)
        
                        if m + 1 == self.CurrentlyViewedSimulation or self.CurrentlyViewedSimulation == 0:
                            C.display()
                            if self.showMovement:
                                C.displayMovement()
        
                        C.Energy += self.RewardFunction(pygame.Vector2.distance_to(pygame.Vector2(CENTREWIDTH, CENTREHEIGHT), C.pos)) - (len(CreaturesList[m]) * CREATUREENERGYCOST)
 
                        if C.pos.x > WIDTH - C.width or C.pos.x < C.width or C.pos.y > HEIGHT - C.height or C.pos.y < C.height or C.TimeAlive >= MAXTIME - (len(CreaturesList[m]) * CREATURETIMEOUT) or C.Energy <= 0:
                            CreaturesList[m].remove(C)
                        
                        if C.Energy >= C.MaxEnergy:
                            C.Energy = C.MaxEnergy/5
                            NewC = Creature(random.randint(CENTREWIDTH - SPAWNDISTANCE, CENTREWIDTH + SPAWNDISTANCE),random.randint(CENTREHEIGHT - SPAWNDISTANCE, CENTREHEIGHT + SPAWNDISTANCE), C.width, C.height, C.MaxSpeed, C.MaxEnergy, C.color)
                            NewC.Generation = C.Generation + 1
                            NewC.InheritNetwork(C)
                            NewC.Mutate(1,6) #0.6% mutation chance
                            NewC.angle = random.randint(-180,180)
                            CreaturesList[m].append(NewC)
                            if self.selectedCreature == None:
                                if self.CurrentlyViewedSimulation - 1 == m or self.CurrentlyViewedSimulation == 0:
                                    self.SetSelectedCreature(NewC)
                self.TotalTime += 1
 
            DrawSidebar()
 
            DrawText(sidebarSurface, f"FPS:      {int(clock.get_fps())}", RED, (0,0))
            DrawText(sidebarSurface, f"Time:    {self.TotalTime}", MAGENTA, (0,20))
            DrawText(sidebarSurface, f"Viewed: {self.CurrentlyViewedSimulation}", CYAN, (0,40))
            RenderSidebar()
 
            if self.showLineGraphs:
                RenderLineGraphs()
 
            if self.selectedCreature != None :
                foundCreature = False
                for m in range(len(CreaturesList)):
                    if self.selectedCreature in CreaturesList[m]:
                        foundCreature = True
                
                selectedInputs = [1, self.selectedCreature.angle / 180, self.selectedCreature.velocity / self.selectedCreature.MaxSpeed, self.selectedCreature.TimeAlive / MAXTIME, (self.selectedCreature.pos.x - CENTREWIDTH) / CENTREWIDTH, (self.selectedCreature.pos.y - CENTREHEIGHT) / CENTREHEIGHT]
                if foundCreature == True:
                    if self.showInfo == True:
                        pygame.draw.rect(screen, YELLOW, (self.selectedCreature.pos.x - 2, self.selectedCreature.pos.y - 2, self.selectedCreature.width + 4, self.selectedCreature.height + 4), 2)
 
                        infoSurface.fill((35,35,35))
                        self.cInfo.Update()
                        RenderInfo()
 
                    if self.showNNDiagram:
                        nnDiagramSurface.fill(LIGHTGREY)
                        if self.showDetail == True:
                            self.nnd.drawWeights()
                            self.nnd.DrawNodes(selectedInputs, 122, 25, 50, True)
                        else:
                            self.nnd.drawWeightsColored(selectedInputs)
                            self.nnd.DrawNodes(selectedInputs, 122, 25, 50, False)
 
                        RenderNNDiagram()
 
                else:
                    self.selectedCreature = None
                    self.nnd.layers = None
 
            pygame.display.update()
            clock.tick(self.fps)
        
MainSim = Game()
 

MainSim.Update()