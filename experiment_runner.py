import gc
import numpy
import random
import numpy as np
from pdb import set_trace as bb
from functools import reduce
import os
from colors import *
import argparse

parser = argparse.ArgumentParser()
#all command line options
parser.add_argument("--display", help="turn on rendering", action="store_true")
parser.add_argument("--mutation", help="whether to use regular mutations or SM",choices=['regular','SM-G-SUM','SM-G-ABS','SM-R','SM-G-SO'],default='regular')
parser.add_argument("--mutation_mag", help="magnitude of mutation operator",default=0.01)
parser.add_argument('--pop_size', help="population size",default=250)
parser.add_argument("--save", help="output file prefix for saving",default="out")
parser.add_argument("--hidden", help="number of hidden units per ann layer", default=15)
parser.add_argument("--init", help="init rule", default="xavier")
parser.add_argument("--celltype", help="recurrent cell type",default="lstm",choices=['lstm','gru','rnn'])
parser.add_argument("--layers", help="number of ann hidden layers",default=6)
parser.add_argument("--activation",help="ann activation function",default="relu")
parser.add_argument("--max_evals",help="total number of evaluations",default=100000)
parser.add_argument("--domain",help="Experimental domain", default="classification",choices=['classification','breadcrumb_maze'])
parser.add_argument("--frameskip",help="frameskip amount (i.e. query agent every X frames for action)", default="3")

#Parse arguments
args = parser.parse_args()
print(args)

#domain selection (whether the recurrent parity task or the breadcrumb hard maze)
if args.domain=='classification':    
    import recurrent_domain as evolution_domain
elif args.domain=='breadcrumb_maze':
    import maze_domain as evolution_domain

#pop up rendering display (for breadcrumb maze domain)
do_display = args.display

#interval for rendering
interval = 50

#make save directory 
os.system("mkdir -p %s" % args.save)

#define dictionary describing ann
params = {'size':int(args.hidden),'af':args.activation,'layers':int(args.layers),'init':args.init,'celltype':args.celltype} 

#define dictionary describing domain
domain = {'name':args.domain,'difference_frames':False,'frameskip':int(args.frameskip),'history':1,'rgb':False,'incentive':'fitness'}

#initialize domain
evolution_domain.setup(domain,params)

#fire up pygame for visualization
import pygame
from pygame.locals import *

#pygame only required for hard maze visualization
pygame.init()
SZX = SZY = 500
screen = None

if do_display:
    pygame.display.set_caption('Viz')
    screen = pygame.display.set_mode((SZX, SZY))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))
else:
    screen = pygame.Surface((SZX, SZY))
    background = pygame.Surface(screen.get_size())
    background.fill((250, 250, 250))
    

#Maze rendering call (visualize behavior of population)
def render_maze(pop):
    global screen, background
    screen.blit(background, (0, 0))

    for robot in pop:
        x = robot.behavior[-2]  #*SZX
        y = robot.behavior[-1]  #*SZY
        rect = (int(x), int(y), 5, 5)
        pygame.draw.rect(screen, (255, 0, 0), rect, 0)

    lines = evolution_domain.individual.env.orig_e.get_line_count()

    for idx in range(lines):
        line = evolution_domain.individual.env.orig_e.get_line(idx)
        start_pos = int(line.a.x), int(line.a.y)
        end_pos = int(line.b.x), int(line.b.y)
        pygame.draw.line(screen, (0, 0, 0), start_pos, end_pos, 2)

    if do_display:
        pygame.display.flip()


if (__name__ == '__main__'):

    #initialize empty population
    population = []

    #placeholders to hold champion
    best_fit = -1e9
    best_ind = None
    best_beh = None

    #grab population size
    psize = int(args.pop_size)

    #initialize population
    for k in range(psize):
        robot = evolution_domain.individual()

        #initialize random parameter vector
        robot.init_rand()

        #evaluate in domain
        robot.map()
        robot.parent = None

        #add to population
        population.append(robot)

    #solution flag
    solved = False

    #we spent evals looking at the population
    evals = psize

    #parse max evaluations
    max_evals = int(args.max_evals)

    #tournament size
    greediness = 5

    #parse mutation intensity parameter
    mutation_mag = float(args.mutation_mag)

    #evolutionary loop
    while evals < max_evals and not solved:
        evals += 1

        if evals % 50 == 0:
            gc.collect()

        if evals % 500 == 0:
            #logging progress to text file
            print("saving out...",evals)
            f = "%s.progress"%args.save

            outline = str(evals)+" "+str(best_fit)

            #write out addtl info if we have it
            if len(best_beh)>0:
                outline = outline+" "+str(best_beh[0])
    
            open(f,"a+").write(outline+"\n")
            f2 = "%s_best.npy" % args.save
            best_ind.save(f2)

        if (evals % interval == 0) and args.domain=="breadcrumb_maze":
            render_maze(population)

        #tournament selection
        parents = random.sample(population, greediness)
        parent = reduce(lambda x, y: x if x.fitness > y.fitness else y,
                        parents)

        #copy parameter vector
        child = parent.copy()
        #mutate
        child.mutate(mutation=args.mutation,mag=mutation_mag) 
        #evalute in domain
        child.map()

        population.append(child)

        print child.fitness
        if child.reward > best_fit:
            best_fit = child.reward
            best_ind = child.copy()
            best_beh = child.behavior[:]

            print bcolors.WARNING
            print "new best fit: ", best_fit, child.behavior
            print bcolors.ENDC

        if (child.solution()):
            solved = True

        if (evals % 100 == 0 or solved):
            idx = 0
            save_all = False
            if save_all:
                for k in population:
                    k.save("%s/child%d" % (args.save,idx))
                    idx += 1

        #remove individual from the pop using a tournament of same size
        to_kill = random.sample(population, greediness)
        to_kill = reduce(lambda x, y: x if x.fitness < y.fitness else y,
                         parents)
        population.remove(to_kill)
        to_kill.kill()
        del to_kill

    print("SOLVED!")
    fname = args.save + "_EVAL%d" % evals
    child.save(fname)
    print("saved locally")

    
