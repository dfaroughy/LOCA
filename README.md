# LOCA

**L**HC**O** **C**utflow **A**nalysis: python module for analyzing collider events in LHCO format.

Main function is: ```open_loca(<file>,<cuts>)```
This opens the lhco files and extracts the event blocks from the sample
that satisfy the basic kinematic selection ```<cuts>```.

# usage:
```
with open_loca(<lhcofile>, <cuts>) as sample:
    for event in sample:
        # apply cuts to event
```
```event``` has several attributes:

- ```event.<objects>``` where ```<objects> = photons, electrons, muons, taus, jets, bjets, met, leptons, all_objects```: list of selected objects of given type in event.
- ```event.<nobjects>``` where ```<nobjects> = nphotons, nelecs, nmuons, ntaus, nleps, njets, nbjets```: number of selected objects of given type in event, 
- ```event.<object_leading>``` where ```<object_leading> = photon_leading, electron_leading, ... ```: leading pt object of given type in event.
- ```event.<object_subleading>``` where ```<object_subleading> = photon_subleading, electron_subleading, ...``` sub-leading pt object of given type in event.

- selected objects have a the usual kinematic attributes: ```object.pt, object.eta, object.is_lepton, obj1.DeltaR(obj2) ...```
- the basic selection ```<cuts>``` must be given as a python dictionary: ```cuts = { <object> : [pt_min, pt_max, abs(eta_max)] }```   
