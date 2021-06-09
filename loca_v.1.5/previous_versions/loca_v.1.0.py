import re
import numpy as np
import time
import sys
import glob
import shlex
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

    def __init__(self, file, cuts):
        self.f=open(file,'r')
        self.cuts=cuts

    def __enter__(self):
        events=[]
        for l in self.f:
            line = str.split(l)
            if len(line) > 0 and line == ['0','0','0']:
                break
        
        electrons=[]
        muons=[]
        taus=[]
        qjets=[]
        bjets=[]
        photons=[]
        met=[]

        spectrum_photon_1 = spectrum()
        spectrum_photon_2 = spectrum()
        spectrum_electron_1 = spectrum()
        spectrum_electron_2 = spectrum()
        spectrum_muon_1 = spectrum()
        spectrum_muon_2 = spectrum()
        spectrum_tau_1 = spectrum()
        spectrum_tau_2 = spectrum()
        spectrum_jet_1 = spectrum()
        spectrum_jet_2 = spectrum()
        spectrum_bjet_1 = spectrum()
        spectrum_bjet_2 = spectrum()
        spectrum_met = spectrum()

        for l in self.f:
            
            line = list(map( float, str.split(l)) ) 
            
            if len(line) > 3:

                obj=Object_Reconstruction(line,self.cuts)
                _object_=obj.Extract_Candidates(photons,electrons,muons,taus,qjets,bjets,met)
                
                if _object_.is_last_in_event:
                    
                    event=Event(photons,electrons,muons,taus,qjets,bjets,met)
                    events.append(event)
                    spectrum_photon_1.get_spectra(event.photon_leading)
                    spectrum_photon_2.get_spectra(event.photon_subleading)
                    spectrum_electron_1.get_spectra(event.electron_leading)
                    spectrum_electron_2.get_spectra(event.electron_subleading)
                    spectrum_muon_1.get_spectra(event.muon_leading)
                    spectrum_muon_2.get_spectra(event.muon_subleading)
                    spectrum_tau_1.get_spectra(event.tau_leading)
                    spectrum_tau_2.get_spectra(event.tau_subleading)
                    spectrum_jet_1.get_spectra(event.jet_leading)
                    spectrum_jet_2.get_spectra(event.jet_subleading)
                    spectrum_bjet_1.get_spectra(event.bjet_leading)
                    spectrum_bjet_2.get_spectra(event.bjet_subleading)
                    spectrum_met.get_spectra(event.met)

                    electrons=[]
                    muons=[]
                    taus=[]
                    qjets=[]
                    bjets=[]
                    photons=[];
                    met=[]
    
        fig = plt.figure(figsize=(8,4*13))
        spectrum_electron_1.plot_spectra(r'$e^{\pm}$ leading'     ,fig,1,13)
        spectrum_electron_2.plot_spectra(r'$e^{\pm}$ sub-leading' ,fig,2,13)
        spectrum_muon_1.plot_spectra(r'$\mu^{\pm}$ leading'       ,fig,3,13)
        spectrum_muon_2.plot_spectra(r'$\mu^{\pm}$ sub-leading'   ,fig,4,13)
        spectrum_tau_1.plot_spectra(r'$\tau^{\pm}$ leading'       ,fig,5,13)
        spectrum_tau_2.plot_spectra(r'$\tau^{\pm}$ sub-leading'   ,fig,6,13)
        spectrum_photon_1.plot_spectra(r'$\gamma$ leading'        ,fig,7,13)
        spectrum_photon_2.plot_spectra(r'$\gamma$ subleading'     ,fig,8,13)
        spectrum_jet_1.plot_spectra(r'jet leading'                ,fig,9,13)
        spectrum_jet_2.plot_spectra(r'jet sub-leading'            ,fig,10,13)
        spectrum_bjet_1.plot_spectra(r'b-jet leading'             ,fig,11,13)
        spectrum_bjet_2.plot_spectra(r'b-jet sub-leading'         ,fig,12,13)
        spectrum_met.plot_spectra(r'missing momentum'             ,fig,13,13)
        plt.tight_layout()
        plt.savefig('pre_selection_plots.pdf')
        return events

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

class Object_Reconstruction:

    def __init__(self, line, cuts):

        ID={0:'photon',1:'e',2:'mu',3:'tau',4:'jet',6:'met'}
        self.typ  = ID[int(line[1])]      
        self.eta  = line[2]   
        self.phi  = line[3]     
        self.pt   = line[4]     
        self.ntrk = line[6]    
        self.btag = line[7] 
        self.cuts = cuts
        self.charge = sign(float(self.ntrk))
        self.theta  = 2. * np.arctan(np.exp(-self.eta))
        self.E  = self.pt / np.sin(self.theta)
        self.px = self.E * np.sin(self.theta) * np.cos(self.phi)
        self.py = self.E * np.sin(self.theta) * np.sin(self.phi)
        self.pz = self.E * np.cos(self.theta)
        self.is_photon = (self.typ == 'gamma')
        self.is_electron = (self.typ == 'e')
        self.is_muon = (self.typ == 'mu')
        self.is_tau  = (self.typ == 'tau')
        self.is_jet  = (self.typ == 'jet')  
        self.is_qjet = (self.typ == 'jet' and self.btag!=1)  
        self.is_bjet = (self.typ == 'jet' and self.btag==1)  
        self.is_met  = (self.typ == 'met') 
        self.is_photon_candidate = False
        self.is_electron_candidate = False
        self.is_muon_candidate = False
        self.is_tau_candidate  = False
        self.is_qjet_candidate  = False  
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
        return np.sqrt(self.inv2Mass(obj))

    def MT(self,obj):#.....Transverse mass between two objects            
        return np.sqrt(2*self.pt*obj.pt*(1-np.cos(self.dphi(obj)))) 

    def MTtot(self,objA,objB):#.....Total transverse mass between three objects 
        return np.sqrt( self.MT(objA)**2 + self.MT(objB)**2 + objA.MT(objB)**2)


    def Extract_Candidates(self,photons,electrons,muons,taus,qjets,bjets,met):  
        if self.is_photon:
            if (self.pt>self.cuts[self.typ][0] and self.pt<self.cuts[self.typ][1] and abs(self.eta)<self.cuts[self.typ][2]):
                self.is_photon_candidate=True
                photons.append(self)
        elif self.is_electron:       
            if (self.pt>self.cuts[self.typ][0] and self.pt<self.cuts[self.typ][1] and abs(self.eta)<self.cuts[self.typ][2]):
                if (abs(self.eta)<1.37 or abs(self.eta)>1.52): # remove endcaps
                    self.is_electron_candidate=True
                    electrons.append(self)
        elif self.is_muon:
            if (self.pt>self.cuts[self.typ][0] and self.pt<self.cuts[self.typ][1] and abs(self.eta)<self.cuts[self.typ][2]):
                self.is_muon_candidate=True
                muons.append(self)
        elif self.is_tau:
            if (self.pt>self.cuts[self.typ][0] and self.pt<self.cuts[self.typ][1] and abs(self.eta)<self.cuts[self.typ][2]):
                if (abs(self.eta)<1.37 or abs(self.eta)>1.52): # endcaps
                    if (abs(int(self.ntrk))==1 or abs(int(self.ntrk))==3):
                        self.is_tau_candidate=True
                        taus.append(self)
        elif self.is_qjet:
            if (self.pt>self.cuts[self.typ][0] and self.pt<self.cuts[self.typ][1] and abs(self.eta)<self.cuts[self.typ][2]):
                self.is_qjet_candidate=True
                qjets.append(self)
        elif self.is_bjet:
            if (self.pt>self.cuts['bjet'][0] and self.pt<self.cuts['bjet'][1] and abs(self.eta)<self.cuts['bjet'][2]):
                self.is_bjet_candidate=True
                self.typ='bjet'
                bjets.append(self)
        elif self.is_met:
            self.is_last_in_event=True
            met.append(self)

        return self

class Event:

    def __init__(self,photons,electrons,muons,taus,qjets,bjets,met):

        photons.sort(key=lambda x: -x.pt)
        electrons.sort(key=lambda x: -x.pt)
        muons.sort(key=lambda x: -x.pt)
        taus.sort(key=lambda x: -x.pt)
        qjets.sort(key=lambda x: -x.pt)
        bjets.sort(key=lambda x: -x.pt)
        leptons = electrons + muons
        jets = qjets + bjets
        visible = photons + leptons + taus + jets
        leptons.sort(key=lambda x: -x.pt)
        jets.sort(key=lambda x: -x.pt)
        visible.sort(key=lambda x: -x.pt)
        
        self.all_objects = visible + met
        self.photons = photons
        self.electrons = electrons
        self.muons = muons
        self.taus = taus
        self.leptons = electrons + muons
        self.qjets = qjets
        self.bjets = bjets
        self.jets = qjets + bjets
        self.met = met[0] 

        self.nelecs = len(electrons)
        self.nmuons = len(muons)
        self.ntaus  = len(taus)
        self.nleps = len(leptons)
        self.njets = len(jets)
        self.nqjets = len(qjets)
        self.nbjets = len(bjets)
        self.nphotons = len(photons)

        if self.nelecs>0:
            self.electron_leading=electrons[0]
        else:
            self.electron_leading=False
        if self.nelecs>1:
            self.electron_subleading=electrons[1]
        else:
            self.electron_subleading=False
        if self.nmuons>0:
            self.muon_leading=muons[0]
        else:
            self.muon_leading=False
        if self.nmuons>1:
            self.muon_subleading=muons[1]
        else:
            self.muon_subleading=False
        if self.ntaus>0:
            self.tau_leading=taus[0]
        else:
            self.tau_leading=False
        if self.ntaus>1:
            self.tau_subleading=taus[1]
        else:
            self.tau_subleading=False
        if self.nleps>0:
            self.lepton_leading=leptons[0]
        else:
            self.lepton_leading=False
        if self.nleps>1:
            self.lepton_subleading=leptons[1]
        else:
            self.lepton_subleading=False
        if self.njets>0:
            self.jet_leading=jets[0]
        else:
            self.jet_leading=False
        if self.njets>1:
            self.jet_subleading=jets[1]
        else:
            self.jet_subleading=False
        if self.nbjets>0:
            self.bjet_leading=bjets[0]
        else:
            self.bjet_leading=False
        if self.nbjets>1:
            self.bjet_subleading=bjets[1]
        else:
            self.bjet_subleading=False
        if self.nphotons>0:
            self.photon_leading=photons[0]
        else:
            self.photon_leading=False
        if self.nphotons>1:
            self.photon_subleading=photons[1]
        else:
            self.photon_subleading=False

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

        if len(self.eta)>10:
            bins_eta=bayesian_blocks(self.eta)
        else:
            bins_eta=10

        ax=fig.add_subplot(nplt,2,2*nrow-1)
        self.pt_plot=plt.hist(self.pt,bins=bins_pt,facecolor='r',alpha=0.5,density=False)
        plt.title(string)
        plt.xlabel(r'$p_t$ [GeV]')
        plt.ylabel('MC Events')
        ax.grid(linestyle='--',linewidth=1)

        ax=fig.add_subplot(nplt,2,nrow*2)
        self.eta_plot=plt.hist(self.eta,bins=bins_eta,facecolor='r',alpha=0.5,density=False)
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
        print('================== Results ====================')
        if process:
            print('process: {}'.format(process))
        if description:
            print('description: {}'.format(description))
            print('===============================================')
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
            RESULTS = open(self.category.name+'_results.dat',"w")
            RESULTS.write('================== Results ====================\n')
            if process:
                RESULTS.write('process: {}\n'.format(process))
            if description:
                RESULTS.write('description: {}\n'.format(description))
                RESULTS.write('===============================================\n')
            RESULTS.write('category: {}\n'.format(self.category.name))
            RESULTS.write('num events = {}\n'.format(int(self.category.N)))
            RESULTS.write('cutflow:\n')
            flow=''
            for cut in self.category.get_cuts:
                Ncut=self.category.apply_cut(cut,False)
                flow+='-'
                RESULTS.write('{} {} : {} ({})\n'.format(flow,cut,int(Ncut),Ncut/self.category.N))
            RESULTS.write('===============================================\n')
            RESULTS.close()
#........................................................

def make_dir(path_new_dir, overwrite=False):
    if overwrite:
        os.mkdir(path_new_dir)
    else:
        for I in it.count():
            Dir=path_new_dir + '__' + str(I)
            if not os.path.isdir(Dir):
                os.mkdir(Dir)
                break
            else:
                continue
    return Dir

#........................................................

def elapsed_time(t0):
    t=timer()-t0
    if t < 60.: 
        res='time: {} sec'.format(t)
    elif (t > 60. and t < 3600.0): 
        res='time: {} min'.format(t/60.)
    elif  t >= 3600.0: 
        res='time: {} hours'.format(t/3600.)
    return res
