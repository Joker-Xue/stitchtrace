"""Complementary descriptive labels, never assembly-causation or clinical accuracy."""
from pathlib import Path
import json
import numpy as np
from skimage.io import imread
from skimage.measure import label
from skimage.metrics import contingency_table
from sklearn.metrics import adjusted_rand_score
R=Path(__file__).resolve().parents[1];rows=[]
for item in json.loads((R/'research/selected-inputs.json').read_text()):
 mask=imread(R/item['mask']['path']);truth=label(mask[:,:,0]);pred=np.load(R/'results/engineering512'/item['id']/'native/output.npy')
 np.save(R/'data/selected'/item['id']/'decoded-reference.npy',truth.astype(np.uint32))
 table=contingency_table(truth,pred,ignore_labels=(0,),normalize=False).tocsr()[:,1:]
 table.eliminate_zeros()  # Sparse stored zeros from ignored background are not overlaps.
 rows.append({'id':item['id'],'ground_truth_objects':int(truth.max()),'native_objects':int(np.count_nonzero(np.unique(pred))),
 'foreground_dice':float(2*np.count_nonzero((truth>0)&(pred>0))/(np.count_nonzero(truth)+np.count_nonzero(pred))),
 'adjusted_rand_all_pixels':float(adjusted_rand_score(truth.ravel(),pred.ravel())),
 'overlap_split_gt_objects':int(np.count_nonzero(table.getnnz(axis=1)>1)),
 'overlap_merge_output_objects':int(np.count_nonzero(table.getnnz(axis=0)>1)),
 'decode':'first PNG channel, skimage.measure.label default connectivity per official example',
 'interpretation':'descriptive model/reference disagreement; not attributed to assembly; training-set overlap not excluded'})
(R/'results/reference-metrics.json').write_text(json.dumps(rows,indent=2));print(rows)
