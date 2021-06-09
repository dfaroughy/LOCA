
# lhco_reader_v.1.2

import re
import numpy as np
from timeit import default_timer as timer
import sys
import glob
import shlex
import shutil
import os.path
import itertools as it
from numpy import sign
import gzip
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from operator import itemgetter
from hepstats.modeling import bayesian_blocks

#
#
#
#
#

class lhco_open(object):

    def __init__(self, file, cuts, lhco_header=False):
        self.f=open(file,'r')
        self.cuts=cuts
        self.parameters={}

        for l in self.f:
            line = str.split(l)

            if len(line) > 0 and line == ['0','0','0']:
                break
            elif 'Integrated weight (pb)  :' in l:
                self.parameters['xsec']=1000.0*float(line[-1])
            elif '#      900000' in l:
                self.parameters[line[-1]]=float(line[2])


        if lhco_header:
            param_card=open('run_parameters','w')
            for p in list(self.parameters.keys()):
                param_card.write('{}\t{}\n'.format(p,self.parameters[p]))
            param_card.close()  

    def __enter__(self):
        photons=[]
        electrons=[]
        muons=[]
        taus=[]
        jets=[]
        bjets=[]
        met=[]
        for l in self.f:
            self.line = list(map( float, str.split(l)) ) 
            if len(self.line)>3:
                obj = Object_Reconstruction(self.line)            
                if obj.is_bjet:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        obj.typ='bjet'
                        bjets.append(obj)
                if obj.is_photon:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        photons.append(obj)
                elif obj.is_electron:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        if (abs(obj.eta)<1.37 or abs(obj.eta)>1.52):  # remove endcaps      
                            electrons.append(obj)
                elif obj.is_muon:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        muons.append(obj)
                elif obj.is_tau:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        if (abs(obj.ntrk)==1 or abs(obj.ntrk)==3): # 1, 3 prongs
                            taus.append(obj)
                elif obj.is_jet:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        obj.typ='jet'
                        jets.append(obj)
                elif obj.is_met:
                    if (obj.pt>self.cuts[obj.typ][0] and obj.pt<self.cuts[obj.typ][1] and abs(obj.eta)<self.cuts[obj.typ][2]):
                        met.append(obj)
                        event=Event(photons,electrons,muons,taus,jets,bjets,met)
                        photons=[]
                        electrons=[]
                        muons=[]
                        taus=[]
                        jets=[]
                        bjets=[]
                        met=[]
                        yield event

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
 
#
#
#
#
#

class Object_Reconstruction:

    def __init__(self, line):

        ID={0:'photon',1:'e',2:'mu',3:'tau',4:'jet',6:'met'}

        self.typ  = ID[int(line[1])]      
        self.eta  = line[2]   
        self.phi  = line[3]     
        self.pt   = line[4]     
        self.ntrk = line[6]    
        self.btag = line[7] 

        self.charge = sign(float(self.ntrk))
        self.theta  = 2. * np.arctan(np.exp(-self.eta))
        self.E  = self.pt / np.sin(self.theta)
        self.px = self.E * np.sin(self.theta) * np.cos(self.phi)
        self.py = self.E * np.sin(self.theta) * np.sin(self.phi)
        self.pz = self.E * np.cos(self.theta) 

        self.is_photon = (self.typ == 'photon')
        self.is_electron = (self.typ == 'e')
        self.is_muon = (self.typ == 'mu')
        self.is_lepton = (self.typ == 'e' or self.typ == 'mu')
        self.is_tau  = (self.typ == 'tau')
        self.is_jet  = (self.typ == 'jet')  
        self.is_bjet = (self.typ == 'jet' and self.btag==1) or (self.typ=='bjet')
        self.is_met  = (self.typ == 'met') 
        # self.is_last_in_event = False

    def dphi(self,obj):  
        dphi = self.phi - obj.phi
        if dphi > np.pi:
            dphi -= 2*np.pi  
        if dphi < -np.pi:
            dphi += 2*np.pi 
        return dphi 

    def deltaR2(self,obj):  
        dphi = self.dphi(obj)
        deta = self.eta - obj.eta
        return deta*deta+dphi*dphi 

    def deltaR(self,obj):  
        return np.sqrt(self.deltaR2(obj)) 

    def invM2(self,obj):
        return (self.E+obj.E)**2-(self.px+obj.px)**2-(self.py+obj.py)**2-(self.pz+obj.pz)**2

    def invM(self,obj):
        return np.sqrt(self.invM2(obj))

    def MT(self,obj):#.....Transverse mass between two objects            
        return np.sqrt(2*self.pt*obj.pt*(1-np.cos(self.dphi(obj)))) 

    def MTtot(self,objA,objB):#.....Total transverse mass between three objects 
        return np.sqrt( self.MT(objA)**2 + self.MT(objB)**2 + objA.MT(objB)**2)

class Event:

    def __init__(self,photons,electrons,muons,taus,jets,bjets,met):

        self.photons=photons
        self.electrons=electrons
        self.muons=muons
        self.leptons=electrons+muons
        self.taus=taus
        self.jets=jets
        self.bjets=bjets
        self.visible=photons+electrons+muons+taus+jets
        self.all_objects=photons+electrons+muons+taus+jets+met
        self.met=met[0]

        self.leptons.sort(key=lambda x: -x.pt)
        self.visible.sort(key=lambda x: -x.pt)
        self.all_objects.sort(key=lambda x: -x.pt)

        self.nphotons=len(self.leptons)
        self.nelecs=len(self.electrons)
        self.nmuons=len(self.muons)
        self.nleps=len(self.leptons)
        self.ntaus=len(self.taus)
        self.njets=len(self.jets)
        self.nbjets=len(self.bjets)

        self.photon_leading=leading(self.photons,0)
        self.photon_subleading=leading(self.photons,1)
        self.electron_leading=leading(self.electrons,0)
        self.electron_subleading=leading(self.electrons,1)
        self.muon_leading=leading(self.muons,0)
        self.muon_subleading=leading(self.muons,1)
        self.lepton_leading=leading(self.leptons,0)
        self.lepton_subleading=leading(self.leptons,1)
        self.tau_leading=leading(self.taus,0)
        self.tau_subleading=leading(self.taus,1)
        self.jet_leading=leading(self.jets,0)
        self.jet_subleading=leading(self.jets,1)
        self.bjet_leading=leading(self.bjets,0)
        self.bjet_subleading=leading(self.bjets,1)

    def print_selected_objects(self, event_number, save_file=False):

        if save_file!=False:
            save_file.write('\n')
            save_file.write('event # {}\n'.format(int(event_number)))
            save_file.write('------------------------------------------------------\n')
            q={1:'+', -1:'-', 0:''}
            for o in self.all_objects:
                if (o.is_lepton or o.is_tau):
                    save_file.write('{}{},\tpt = {},\teta = {},\tphi = {}\n'.format(o.typ,q[int(o.charge)],o.pt,o.eta,o.phi)) 
                elif o.is_bjet:
                    save_file.write('{},\tpt = {},\teta = {},\tphi = {}\n'.format('bjet',o.pt,o.eta,o.phi))
                else: 
                   save_file.write('{},\tpt = {},\teta = {},\tphi = {}\n'.format(o.typ,o.pt,o.eta,o.phi))
            save_file.write('------------------------------------------------------')

        else:
            print('\n')
            print('event # {}'.format(int(event_number)))
            print('------------------------------------------------------')
            q={1:'+', -1:'-', 0:''}
            for o in self.all_objects:
                if (o.is_lepton or o.is_tau):
                    print('{}{},\tpt = {},\teta = {},\tphi = {}'.format(o.typ,q[int(o.charge)],o.pt,o.eta,o.phi))
                elif o.is_bjet:
                    print('{},\tpt = {},\teta = {},\tphi = {}'.format('bjet',o.pt,o.eta,o.phi))
                else: 
                    print('{},\tpt = {},\teta = {},\tphi = {}'.format(o.typ,o.pt,o.eta,o.phi))
            print('------------------------------------------------------')

#
#
#
#
#

class spectrum:

    def __init__(self,name):
        self.spectrum=[]
        self.name=name

    def extract_spectrum(self, observable):
        self.spectrum.append(observable)

    def bin_data(self,bins,save_file=False):
        count, _ = np.histogram(self.spectrum, bins)

        if save_file:
            file=open('{}_spectrum.dat'.format(self.name),'w')
            file.write('# {} spectrum counts: \n'.format(self.name))
            for c in count:
                file.write(str(c)+'\t')
            file.write('\n')
            file.close()

        return count

#
#
#
#
#

class basic_spectrum:

    def __init__(self,event=False,name=False,bins=False,save=False):
        self.pt = []
        self.eta = []
        self.pt_plot = 0
        self.eta_plot = 0
        if event==True:
            self.event=event
            self.bins=bins
            self.save=True
            self.counts=np.zeros(len(bins)-1)

    def get_spectra(self,obj):
        if obj!=False:
            self.pt.append(obj.pt)
            self.eta.append(obj.eta)

    # def extract_spectrum(self, observable):

    def plot_spectra(self,string,fig,nrow,nplt):  

        if len(self.pt)>10:
            bins_pt=bayesian_blocks(self.pt)
        else:
            bins_pt=10

        ax=fig.add_subplot(nplt,2,2*nrow-1)
        self.pt_plot=plt.hist(self.pt,bins=bins_pt,facecolor='r',alpha=0.5,density=False)
        plt.title(string)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$p_t$ [GeV]')
        plt.ylabel('MC Events')
        ax.grid(linestyle='--',linewidth=1)

        ax=fig.add_subplot(nplt,2,nrow*2)
        self.eta_plot=plt.hist(self.eta,bins=15,facecolor='b',alpha=0.5,density=False)
        plt.title(string)
        plt.xlabel(r'$\eta$')
        plt.ylabel('MC Events')
        ax.grid(linestyle='--',linewidth=1)

#
#
#
#
#

class define_category:

    def __init__(self, *cuts):
        self.cuts={}
        self.passed_cut={}
        self.get_cuts=cuts
        self.num_cuts=len(cuts)
        for c in cuts:
            self.cuts[c]=0.
            self.passed_cut[c]=False
        self.count = 0.
        self.name=''

    def apply_cut(self,cut,count=True):
        if count:
            self.cuts[cut]+=1
            self.passed_cut[cut]=True
        return self.cuts[cut]

    def start_cutflow(self,name):
        self.count+=1 
        self.name=name
#
#
#
#
#

class search_results():
    
    def __init__(self,category):
        self.category=category

    def print(self,process=False,description=False,save_file=False):
        print('\n')
        print('----- Results ----')
        if process:
            print('process: {}'.format(process))
        if description:
            print('description: {}'.format(description))
            print('-----\n')
        print('category: {}'.format(self.category.name))
        print('num events = {}'.format(int(self.category.count)))
        print('cutflow:')
        flow=''
        for cut in self.category.get_cuts:
            Ncut=self.category.apply_cut(cut,False)
            flow+='-'
            print('{} {} : {} ({})'.format(flow,cut,int(Ncut),Ncut/self.category.count))
        print('\n')

        if save_file:
            RESULTS = open('{}_results.dat'.format(self.category.name),"w")
            if process:
                RESULTS.write('process: {}\n'.format(process))
            if description:
                RESULTS.write('description: {}\n'.format(description))
                RESULTS.write('\n')
            RESULTS.write('category: {}\n'.format(self.category.name))
            RESULTS.write('num events = {}\n'.format(int(self.category.count)))
            RESULTS.write('cutflow:\n')
            flow=''
            for cut in self.category.get_cuts:
                Ncut=self.category.apply_cut(cut,False)
                flow+='-'
                RESULTS.write('{} {} : {} ({})\n'.format(flow,cut,int(Ncut),Ncut/self.category.count))
            RESULTS.close()
#........................................................
#
#
#
#
#
# aux functions:

def leading(objects,n):
    if len(objects)>n:
        objects.sort(key=lambda x: -x.pt)
        return objects[int(n)]
    else:
        return False

def make_dir(path_new_dir, overwrite=False):
    Directory=path_new_dir
    if overwrite:
        shutil.rmtree(Directory, ignore_errors=True)
        os.mkdir(Directory)
    else:
        for I in it.count():
            Directory=path_new_dir + '__' + str(I+1)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)
                break
            else:
                continue
    return Directory

def elapsed_time(t0):
    t=timer()-t0
    if t < 60.: 
        res='time: {} sec'.format(t)
    elif (t > 60. and t < 3600.0): 
        res='time: {} min'.format(t/60.)
    elif  t >= 3600.0: 
        res='time: {} hours'.format(t/3600.)
    return res


