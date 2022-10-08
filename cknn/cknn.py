import numpy as np
import ot

from lda.lda import TensorDecompositionLDA
from textrank.textrank import TextRank

class CKNN( object ):

	def __init__( self, V, costfun ):

		self._V = V
		self._v2idx = { v:idx for idx, v in enumerate( V ) }
		self._f = costfun


	def text2freq( self, text ):

		C = np.zeros( len( self._V ) )

		for word in text.split( ' ' ):
			C[ self._v2idx[ word ] ] += 1

		return C

	def topics( self, X, n, alpha0=1., Phi=None ):

		self.LDA = TensorDecompositionLDA( n, alpha0 )

		if Phi:
			rows, cols = Phi.shape

			if cols != n:
				raise ValueError( 'Phi must be a m x n matrix where m is the number of words in the vocabulary and n is the number of topics' )

			self.LDA.Phi = Phi
		else:

			self.LDA.fit( X )

		return self.LDA


	def neighbours( self, text, k ):

		# convert input document into a frequency matrix
		freq = self.text2freq( text )

		# compute the distribution over topics for the input document
		texttopics = self.LDA.doc_topics( C )

		# get its most likely topic
		z = np.argmax( texttopics[ 'document 0' ] )

		if not queryset:
			# compute the query set by running textrank on the document
			textrank = TextRank()
			queryset = textrank.keywords( text )

		querysetidxs = [ self._v2idx[ q ] for q in queryset ]

		# compute cost matrix
		M = self._f( self._V, queryset )

		# compute the "demand" histogram D
		D = self.LDA.Phi[ :, z ]

		# compute the "capacity" histogram C 
		qcondx = freq[ querysetidxs ] / np.sum( freq )
		C = self.LDA.Phi[ querysetidxs, z ] + qcondx
		C = C / np.sum( C, axis=0 )

		# compute the optimal transport
		Pi = ot.emd( 
			np.ascontiguousarray( M ),
			np.ascontiguousarray( D ),
			np.ascontiguousarray( C )
		)

		# pick the k largest values from Pi
		lowerbound = np.sort( Pi.flatten() )[ ::-1 ][ k ]
		coords = np.where( Pi >= lowerbound )

		# return the k vocabulary words that are semantically
		# closest to the text
		return [ self.V[ x ] for x, y in coords ][ :k ]






