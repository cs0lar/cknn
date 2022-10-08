import unittest
import os

import numpy as np
import torch
import torchtext

from torch.nn import CosineSimilarity 

from cknn.cknn import CKNN

GLOVE_VEC_LEN = 100

glove = torchtext.vocab.GloVe( name="6B", dim=GLOVE_VEC_LEN )

def similarity( X, Y ):

	f = CosineSimilarity( dim=1 )

	vX = torch.vstack( [ glove[ w ] for w in X ] )
	vY = torch.vstack( [ glove[ w ] for w in Y ] )

	return sim( vX, vY )

class TestCKNN( unittest.TestCase ):

	def setUp( self ):
		
		assetpath = os.path.join( os.getcwd(), 'tests/assets' )
		self.V, self.trainset, self.testset = nytdataset( assetpath )


	def testNYTDataset( self ):

		cknn = CKNN( self.V, similarity )

		text = ' '.join( [ self.V[ idx ] for idx, freq in enumerate( self.testset[ :, 0 ] ) if freq > 0 ] )

		print ( text )
		

		cknn.topics( self.trainset, 10 )

		print ( cknn.neighbours( text, 10 ) )




