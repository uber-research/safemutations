import glob

incdir = "/usr/local/lib/python2.7/dist-packages/numpy/core/include/"

env=Environment(CPPFLAGS='-I include/ -I %s -march=native -O2' % incdir)
#env=Environment(CPPFLAGS='-g -I include/')
env.ParseConfig('python2-config --includes --libs')

allsrc=['maze.cpp','mazesim_wrap.cxx'] #glob.glob('*.cpp')+glob.glob('*.cxx')

#env.SharedLibrary(target="_mazesim",source=allsrc,SHLIBPREFIX='',LIBS=['m','python2.7'])
env.SharedLibrary(target="_mazesim",source=allsrc,SHLIBPREFIX='')
