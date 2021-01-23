
# lhco_reader_v.1.1

import re
import numpy as np
import time
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

class lhco_open(object):

    def __init__(self, file, cuts, plots=False):
        self.f=open(file,'r')
        self.cuts=cuts
        self.plots=plots

    def __enter__(self):
        events=[]
        for l in self.f:
            line = str.split(l)
            if len(line) > 0 and line == ['0','0','0']:
                break
        
        photons=[]
        electrons=[]
        muons=[]
        taus=[]
        jets=[]
        bjets=[]
        met=[]

        if self.plots:
            spectrum_lep_1  = spectrum()
            spectrum_lep_2  = spectrum()
            spectrum_tau_1  = spectrum()
            spectrum_tau_2  = spectrum()
            spectrum_jet_1  = spectrum()
            spectrum_jet_2  = spectrum()
            spectrum_bjet_1 = spectrum()
            spectrum_bjet_2 = spectrum()
            spectrum_met    = spectrum()

        Dir = make_dir('Results', overwrite=True)

        for l in self.f:
            
            line = list(map( float, str.split(l)) ) 
            
            if len(line)>3:

                obj = Object_Reconstruction(line)
                _object_ = obj.Extract_Candidates(photons, electrons, muons, taus, jets, bjets, met)
                
                if _object_.is_last_in_event:
                    
                    event=Event(self.cuts,photons,electrons,muons,taus,jets,bjets,met)
                    events.append(event)

                    if self.plots:

                        spectrum_lep_1.get_spectra(leading(event._leptons_,0))
                        spectrum_lep_2.get_spectra(leading(event._leptons_,1))
                        spectrum_tau_1.get_spectra(leading(event._taus_,0))
                        spectrum_tau_2.get_spectra(leading(event._taus_,1))
                        spectrum_jet_1.get_spectra(leading(event._jets_,0))
                        spectrum_jet_2.get_spectra(leading(event._jets_,1))
                        spectrum_bjet_1.get_spectra(leading(event._bjets_,0))
                        spectrum_bjet_2.get_spectra(leading(event._bjets_,1))
                        spectrum_met.get_spectra(event.met)

                    electrons=[]
                    muons=[]
                    taus=[]
                    jets=[]
                    bjets=[]
                    photons=[];
                    met=[]
    
        if self.plots:
            fig = plt.figure(figsize=(8,3.5*15))
            spectrum_lep_1.plot_spectra(r'$\ell^{\pm}$ leading'       ,fig,7,15)
            spectrum_lep_2.plot_spectra(r'$\ell^{\pm}$ sub-leading'   ,fig,8,15)
            spectrum_tau_1.plot_spectra(r'$\tau^{\pm}$ leading'       ,fig,5,15)
            spectrum_tau_2.plot_spectra(r'$\tau^{\pm}$ sub-leading'   ,fig,6,15)
            spectrum_jet_1.plot_spectra(r'jet leading'                ,fig,11,15)
            spectrum_jet_2.plot_spectra(r'jet sub-leading'            ,fig,12,15)
            spectrum_bjet_1.plot_spectra(r'b-jet leading'             ,fig,13,15)
            spectrum_bjet_2.plot_spectra(r'b-jet sub-leading'         ,fig,14,15)
            spectrum_met.plot_spectra(r'missing momentum'             ,fig,15,15)
            plt.tight_layout()
            plt.savefig(Dir+'/pre_selection_plots.pdf')
        return events

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

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
        self.is_photon = (self.typ == 'gamma')
        self.is_electron = (self.typ == 'e')
        self.is_muon = (self.typ == 'mu')
        self.is_lepton = (self.typ == 'e' or self.typ == 'mu')
        self.is_tau  = (self.typ == 'tau')
        self.is_jet  = (self.typ == 'jet')  
        self.is_bjet = (self.typ == 'jet' and self.btag==1) 
        self.is_met  = (self.typ == 'met') 
        self.is_photon_candidate = False
        self.is_electron_candidate = False
        self.is_muon_candidate = False
        self.is_tau_candidate  = False
        self.is_bjet_candidate = False
        self.is_last_in_event  = False

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



    def Extract_Candidates(self,photons,electrons,muons,taus,jets,bjets,met):  
        
        if self.is_photon:
            self.is_photon_candidate=True
            photons.append(self)

        elif self.is_electron:
            if (abs(self.eta)<1.37 or abs(self.eta)>1.52):  # remove endcaps      
                self.is_electron_candidate=True
                electrons.append(self)

        elif self.is_muon:
            self.is_muon_candidate=True
            muons.append(self)

        elif self.is_tau:
            if (abs(self.eta)<1.37 or abs(self.eta)>1.52):  # remove endcaps
                if (abs(int(self.ntrk))==1 or abs(int(self.ntrk))==3): 
                    self.is_tau_candidate=True
                    taus.append(self)

        elif self.is_jet:
            self.is_jet_candidate=True
            self.typ='bjet'
            jets.append(self)

        elif self.is_met:
            self.is_last_in_event=True
            met.append(self)

        if self.is_bjet:
            self.is_bjet_candidate=True
            self.typ='bjet'
            bjets.append(self)

        return self

class Event:

    def __init__(self,cuts,photons,electrons,muons,taus,jets,bjets,met):
        self.cuts = cuts
        self._photons_=photons
        self._electrons_= electrons
        self._muons_= muons
        self._taus_= taus
        self._jets_= jets
        self._bjets_= bjets
        self._leptons_= electrons + muons
        self._visible_= photons + self._leptons_ + taus + jets
        self._all_objects_= self._visible_ + met

        # apply basic cuts:
        self.photons=[x for x in photons if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.electrons=[x for x in electrons if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.muons=[x for x in muons if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.taus=[x for x in taus if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.jets=[x for x in jets if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.bjets=[x for x in bjets if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        MET=[x for x in met if (x.pt>cuts[x.typ][0] and x.pt<cuts[x.typ][1] and abs(x.eta)<cuts[x.typ][2])]
        self.met=MET[0]
        self.leptons = self.electrons + self.muons
        self.leptons.sort(key=lambda x: -x.pt)
        self.visible = self.photons + self.leptons + self.jets + self.taus
        self.visible.sort(key=lambda x: -x.pt)
        self.all_objects = self.visible + MET
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

    def print_selected_objects(self):
        print('---------------------------------')
        for o in self.all_objects:
            print('{}, pt = {}, eta = {}, phi = {}'.format(o.typ,o.pt,o.eta,o.phi))


class spectrum:

    def __init__(self):
        self.pt = []
        self.eta = []
        self.pt_plot = 0
        self.eta_plot = 0

    def get_spectra(self,obj):
        if obj!=False:
            self.pt.append(obj.pt)
            self.eta.append(obj.eta)

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


class define_category:

    def __init__(self, *cuts):
        self.cuts={}
        self.get_cuts=cuts
        self.num_cuts=len(cuts)
        for c in cuts:
            self.cuts[c]=0.
        self.N = 0.
        self.name=''

    def apply_cut(self,cut,count=True):
        if count:
            self.cuts[cut]+=1
        return self.cuts[cut]

    def initialize(self,name):
        self.N+=1 
        self.name=name

class search_results():
    
    def __init__(self,category):
        self.category=category

    def print(self,process=False,description=False,save=False):
        print('\n')
        print('----- Results ----')
        if process:
            print('process: {}'.format(process))
        if description:
            print('description: {}'.format(description))
            print('-----\n')
        print('category: {}'.format(self.category.name))
        print('num events = {}'.format(int(self.category.N)))
        print('cutflow:')
        flow=''
        for cut in self.category.get_cuts:
            Ncut=self.category.apply_cut(cut,False)
            flow+='-'
            print('{} {} : {} ({})'.format(flow,cut,int(Ncut),Ncut/self.category.N))
        print('\n')

        if save:
            RESULTS = open('Results/'+self.category.name+'_results.dat',"w")
            if process:
                RESULTS.write('process: {}\n'.format(process))
            if description:
                RESULTS.write('description: {}\n'.format(description))
                RESULTS.write('\n')
            RESULTS.write('category: {}\n'.format(self.category.name))
            RESULTS.write('num events = {}\n'.format(int(self.category.N)))
            RESULTS.write('cutflow:\n')
            flow=''
            for cut in self.category.get_cuts:
                Ncut=self.category.apply_cut(cut,False)
                flow+='-'
                RESULTS.write('{} {} : {} ({})\n'.format(flow,cut,int(Ncut),Ncut/self.category.N))
            RESULTS.close()
#........................................................


# aux functions:

def leading(objects,n):
    if len(objects)>n:
        objects.sort(key=lambda x: -x.pt)
        return objects[int(n)]
    else:
        return False

def make_dir(path_new_dir, overwrite=False):
    Dir=path_new_dir
    if overwrite:
        shutil.rmtree(Dir)
        os.mkdir(Dir)
    else:
        for I in it.count():
            Dir=path_new_dir + '__' + str(I)
            if not os.path.isdir(Dir):
                os.mkdir(Dir)
                break
            else:
                continue
    return Dir

def elapsed_time(t0):
    t=timer()-t0
    if t < 60.: 
        res='time: {} sec'.format(t)
    elif (t > 60. and t < 3600.0): 
        res='time: {} min'.format(t/60.)
    elif  t >= 3600.0: 
        res='time: {} hours'.format(t/3600.)
    return res


