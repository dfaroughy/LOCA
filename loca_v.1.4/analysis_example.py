import numpy as np
import sys
import os.path
import glob
from matplotlib import pyplot as plt
from loca import *
import mplhep as hep 
from hepstats.modeling import bayesian_blocks

# template

#============================================================================
# inputs:

lhco_file=____

process='p p --> X Y'

search_description=''' bla bla bla '''

# define pre-selection requirements:

cuts = {    'e'     : [0.0 , 1e+10 ,  1e+10],  # [pt_min, pt_max, abs(eta_max)] 
            'mu'    : [0.0 , 1e+10 ,  1e+10],
            'tau'   : [0.0 , 1e+10 ,  1e+10],
            'jet'   : [0.0 , 1e+10 ,  1e+10],
            'bjet'  : [0.0 , 1e+10 ,  1e+10],
            'met'   : [0.0 , 1e+10 ,  1e+10],
            'photon': [0.0 , 1e+10 ,  1e+10] 
        }

# define signal region categories:

SR_1 = category('cut 1', 'cut 2', 'cut 3')
SR_2 = category('cut 1', 'cut 2', 'cut 3', 'cut 4')

# define distributions to be extracted:

spectrum=spectrum(name='my spectrum')

# ============================================================================


# Selection cuts:

with open_loca(file=lhco_file, cuts=cuts) as events:
    for event in events:

#.......SR 1 category:

        SR_1.start_cutflow(name='Signal Region 1')

        if (____):
            SR1.apply_cut('cut 1')
            # do something...

            if (____):
                SR1.apply_cut('cut 2')
                # do something...

                if (____):
                    SR1.apply_cut('cut 3')
                    
                    observable = ____
                    spectrum.extract_spectrum(observable) 

#.......SR 2 category:

        SR_2.start_cutflow(name='Signal Region 2')

        if (____):
            SR2.apply_cut('cut 1')
	
            # do something...

            if (____):
                SR2.apply_cut('cut 2')
		
                # do something...

                if (____):
                    SR2.apply_cut('cut 3')
		
                    # do something...

                    if (____):
                        SR2.apply_cut('cut 4')  
			
                        # do something...          

# Outputs:

search_results(SR1).print(process=process, description=search_description, save_file=False)
search_results(SR2).print(process=process, description=search_description, save_file=False)

bins=[____]

print(spectrum.bin_data(bins, save_file=True))
