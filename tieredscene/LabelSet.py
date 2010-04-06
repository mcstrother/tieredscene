'''
Created on Apr 4, 2010

@author: mcstrother
'''


class LabelSet(object):
    
    def __init__(self, middle_labels, top_label= 'T', bottom_label = 'B' ):
        """Creates a label set with the specified labels.
        
        Parameters
        ----------
        top_label :
        bottom_label :
        middle_labels : a list of arbitrary objects (usually strings)
        
        """
        self._top = top_label
        self._bottom = bottom_label
        self._middle = middle_labels
        
        #set up a dictionary to quickly hash each label to a number
        self._label_number_dict = {self._top : len(self._middle), self._bottom : len(self._middle)+1}
        for i, label in enumerate(self._middle):
            if label in self._label_number_dict:
                raise ValueError("At least one label passed into LabelSet appears more than once.")
            else:
                self._label_number_dict[label] = i
        
        
    @property
    def top(self):
        return self._top
    
    @property
    def bottom(self):
        return self._bottom
    
    @property
    def middle(self):
        return self._middle
    
    def to_int(self,label):
        """Returns a unique int corresponding to the label in this set 
        
        The middle labels are numbered 0 to len(self.middle)-1.  The top
        label is len(self.middle).  The bottom label is len(self.middle)+1
        """
        return self._label_number_dict[label]
    
