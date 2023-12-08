import os
import numpy as np
import pandas as pd
import json
import nibabel as nib

from utils import eval_utils, io

#
#   Segmentation performance measures
#   Image distance measures
#   Landmark distances
#


def evaluation(data_dir, out_dir, config_file, subset='testing'):

    #
    #   Evaluation of image registration
    #

    
    if isinstance(config_file, str):
        with open(config_file, 'r') as f:
            data = json.load(f)
    else:
        data = config_file

    eval_pairs = data[subset]
    N = len(eval_pairs)
    name=data['task_name']

    measures=[tmp['metric'] for tmp in data['evaluation_methods']]
    measures_orig = [tmp['metric'] for tmp in data['evaluation_methods']]
    print(measures)
    measures_labels = {}
    results_labels = {}
    labels = {}
    if 'dice' in measures:
        measures.insert(measures.index('dice'), 'dice_before')
        labels['dice'] = data['evaluation_methods'][measures_orig.index('dice')]['labels']

        # if we have more than one label:
        if len(labels['dice'])>1:
            # store it for each label
            measures_labels['dice'] = []
            for lab in labels['dice']:
                measures_labels['dice'].append(f'dice_{lab}_before')
                measures_labels['dice'].append(f'dice_{lab}')
            
        results_labels['dice'] = None
    if 'hd95' in measures:
        measures.insert(measures.index('hd95'), 'hd95_before')
        labels['hd95'] = data['evaluation_methods'][measures_orig.index('hd95')]['labels']
        if len(labels['hd95'])>1:
            # store it for each label
            measures_labels['hd95'] = []
            for lab in labels['hd95']:
                measures_labels['hd95'].append(f'hd95_{lab}_before')
                measures_labels['hd95'].append(f'hd95_{lab}')
        results_labels['hd95'] = None
    if 'tre' in measures:
        measures.insert(measures.index('tre'), 'tre_before')
        measures.insert(measures.index('tre'), 'tre_before (std)')
        measures.insert(measures.index('tre')+1, 'tre (std)')
        results_labels['tre'] = None
    if 'sdlogj' in measures:
        measures.insert(measures.index('sdlogj')+1, 'minJDet')
        measures.insert(measures.index('sdlogj')+2, 'maxJDet')
        measures.insert(measures.index('sdlogj')+3, 'negJDet%')
        measures.insert(measures.index('sdlogj')+4, 'num_foldings')
        results_labels['sdlogj'] = None

    print(measures)

    if 'masked_evaluation' in data:
        use_mask = data['use_mask']
    else:
        use_mask = False

    results = np.zeros((N,len(measures)))
    cases = []

    for i, pair in enumerate(eval_pairs):

        # if i>0:
            # continue

        # get case directory
        # print(i, pair)
        fixed_id = pair['fixed'].split('_')[-2]
        moving_id = pair['moving'].split('_')[-2]
        if fixed_id==moving_id:
            # intra-patient: same patient for fixed and moving image
            idx = pair['fixed'].replace('./imagesTr/', '').replace('_0000.nii.gz', '')
            case_dir = f'{out_dir}/{idx}'
        else:
            # inter-patient: different patient for fixed and moving image
            idx = f"{pair['fixed'].replace('./imagesTr/', '').replace('_0000.nii.gz', '').replace('_', '-')}-{moving_id}"
            case_dir = f'{out_dir}/{idx}'
        cases.append(idx)

        # load all necessary files

        # ground truth
        fixed_file = f'{case_dir}/fixed_image.nii.gz'
        moving_file = f'{case_dir}/moving_image.nii.gz'
        fixed_mask_file = f'{case_dir}/fixed_mask.nii.gz'
        moving_mask_file = f'{case_dir}/moving_mask.nii.gz'
        fixed_labels_file = f'{case_dir}/fixed_labels.nii.gz'
        moving_labels_file = f'{case_dir}/moving_labels.nii.gz'
        fixed_landmark_file = f'{data_dir}/{pair["fixed"].replace("./images","/keypoints").replace(".nii.gz",".csv")}'
        moving_landmark_file = f'{data_dir}/{pair["moving"].replace("./images","/keypoints").replace(".nii.gz",".csv")}'

        # case specific
        # disp_file = f'{out_dir}/displ_field.nii.gz'
        fixed_landmarks_warped_file = f'{case_dir}/fixed_landmarks_warped.csv'
        moving_mask_warped_file = f'{case_dir}/moving_mask_warped.nii.gz'
        moving_labels_warped_file = f'{case_dir}/moving_labels_warped.nii.gz'
        disp_file = f'{case_dir}/displ_field.nii.gz'

        # if not os.path.exists(disp_file):
            # continue
         

        # Load spacing og fixed and moving image
        if any([True for eval_ in ['tre'] if eval_ in measures]):
            spacing_fix = [1.5, 1.5, 1.5]
            spacing_mov = [1.5, 1.5, 1.5]

        # load the masks
        if any([True for eval_ in ['dice','hd95'] if eval_ in measures]):
            evaluate_mask = False
            evaluate_labels = False
            # check if we have image masks
            if os.path.exists(fixed_mask_file):
                evaluate_mask = True
                fixed_mask = nib.load(fixed_mask_file).get_fdata()
                moving_mask = nib.load(moving_mask_file).get_fdata()
                moving_mask_warped = np.where(nib.load(moving_mask_warped_file).get_fdata()>0.5, 1, 0)
                mask = fixed_mask
            
            # check if we have organ labels
            if os.path.exists(fixed_labels_file):
                evaluate_labels = True
                fixed_labels = nib.load(fixed_labels_file).get_fdata()
                moving_labels = nib.load(moving_labels_file).get_fdata()
                # moving_labels_warped = np.where(nib.load(moving_labels_warped_file).get_fdata()>0.5, 1, 0)
                moving_labels_warped = nib.load(moving_labels_warped_file).get_fdata()

        if any([True for eval_ in ['sdlogj'] if eval_ in measures]):
            disp_field = nib.load(disp_file).get_fdata()

        string_before = 'Before: \t'
        string_after = 'After: \t\t'
        use_tre, use_dice, use_hd95 = False, False, False
        ## iterate over designated evaluation metrics
        for _eval in data['evaluation_methods']:
            _name=_eval['name']

            # print(_name)

            ### TRE
            if 'tre' == _eval['metric']:
                destination = _eval['dest']
                use_tre = True
                fix_lms = np.loadtxt(fixed_landmark_file, delimiter=',')
                mov_lms = np.loadtxt(moving_landmark_file, delimiter=',')
                fix_lms_warped = np.loadtxt(fixed_landmarks_warped_file, delimiter=',')
                
                tre_before = eval_utils.compute_tre(fix_lms, mov_lms, spacing_fix, spacing_mov, fix_lms_warped=fix_lms, disp=None)
                tre = eval_utils.compute_tre(fix_lms, mov_lms, spacing_fix, spacing_mov, fix_lms_warped=fix_lms_warped, disp=None)
                
                # tre_before2 = eval_utils.compute_landmark_accuracy(fix_lms, mov_lms, spacing_fix)
                # tre2 = eval_utils.compute_landmark_accuracy(fix_lms_warped, mov_lms, spacing_fix)

                # print(tre.mean(), tre.std())
                # print(tre_before.mean(), tre_before.std())
                results[i,measures.index('tre')] = tre.mean()
                results[i,measures.index('tre_before')] = tre_before.mean()
                results[i,measures.index('tre (std)')] = tre.std()
                results[i,measures.index('tre_before (std)')] = tre_before.std()

                string_before += f'TRE {tre_before.mean():.3f} +/- {tre_before.std():.3f}\t'
                string_after += f'TRE {tre.mean():.3f} +/- {tre.std():.3f}\t'


            ### DSC
            if 'dice' == _eval['metric']:
                labels = _eval['labels']
                # print(labels)
                use_dice = True

                if evaluate_mask:
                    dice_before = eval_utils.compute_dice(fixed_mask,moving_mask,moving_mask,labels)
                    dice = eval_utils.compute_dice(fixed_mask,moving_mask,moving_mask_warped,labels)
                elif evaluate_labels:
                    dice_before = eval_utils.compute_dice(fixed_labels,moving_labels,moving_labels,labels)
                    dice = eval_utils.compute_dice(fixed_labels,moving_labels,moving_labels_warped,labels)

                print()
                print("DICE")

                results[i,measures.index('dice')] = dice[0]
                results[i,measures.index('dice_before')] = dice_before[0]

                string_before += f'Dice {dice_before[0]:.3f}\t'
                string_after += f'Dice {dice[0]:.3f}\t'

                # print(string_before)
                # print(string_after)

                if len(labels)>1:
                    # store it for each label
                    if results_labels['dice'] is None:
                        results_labels['dice'] = np.zeros((N,2*len(labels)))
                        results_labels['dice'][0,:] = np.ravel([dice_before[1], dice[1]], 'F')
                    else:
                        results_labels['dice'][i,:] = np.expand_dims(np.ravel([dice_before[1], dice[1]], 'F'),axis=0)

            ### HD95
            if 'hd95' == _eval['metric']:
                labels = _eval['labels']
                use_hd95 = True

                if evaluate_mask:
                    hd95_before = eval_utils.compute_hd95(fixed_mask,moving_mask,moving_mask,labels)
                    hd95 = eval_utils.compute_hd95(fixed_mask,moving_mask,moving_mask_warped,labels)
                elif evaluate_labels:
                    hd95_before = eval_utils.compute_hd95(fixed_labels,moving_labels,moving_labels,labels)
                    hd95 = eval_utils.compute_hd95(fixed_labels,moving_labels,moving_labels_warped,labels)


                results[i,measures.index('hd95')] = hd95[0]
                results[i,measures.index('hd95_before')] = hd95_before[0]

                string_before += f'HD95 {hd95_before[0]:.3f}\t'
                string_after += f'HD95 {hd95[0]:.3f}\t'

                if len(labels)>1:
                    # store it for each label

                    if results_labels['hd95'] is None:
                        results_labels['hd95'] = np.zeros((N,2*len(labels)))
                        results_labels['hd95'][0,:] = np.ravel([hd95_before[1], hd95[1]], 'F')
                    else:
                        results_labels['hd95'][i,:] = np.expand_dims(np.ravel([hd95_before[1], hd95[1]], 'F'),axis=0)


            ### SDlogJ
            if 'sdlogj' == _eval['metric']:
                use_slog = True
                jac_det = (eval_utils.jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose((0,4,1,2,3))) + 3).clip(0.000000001, 1000000000)
                jac_det_noclip = (eval_utils.jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose((0,4,1,2,3))) + 3)
                
                log_jac_det = np.log(jac_det)
                if use_mask:# and mask_ready:
                    sdlogj=np.ma.MaskedArray(log_jac_det, 1-mask[2:-2, 2:-2, 2:-2]).std()
                else:
                    sdlogj=log_jac_det.std()
                num_foldings=(jac_det <= 0).astype(float).sum()

                # jac_det_noclip[100,100,100] = -1
                # jac_det_noclip[100:200,100:150,100:120] = -1
                min_jac_det = jac_det_noclip.min()
                max_jac_det = jac_det_noclip.max()
                perc_neg_jac_det = (jac_det_noclip <= 0).astype(float).sum()*100/np.prod(jac_det_noclip.shape)

                string_after += f'SDLogJ {sdlogj:.5f} \t MinJDet {min_jac_det:.5f} \t MaxJDet {max_jac_det:.5f} \t %NegJDet {perc_neg_jac_det:.5f} \t # foldings {num_foldings:.5f}\t'

                results[i,measures.index('sdlogj')] = sdlogj
                results[i,measures.index('minJDet')] = min_jac_det
                results[i,measures.index('maxJDet')] = max_jac_det
                results[i,measures.index('negJDet%')] = perc_neg_jac_det
                results[i,measures.index('num_foldings')] = num_foldings

        print(f'{idx}:')
        print(string_before)
        print(string_after)
        print()

    if N>1:
        results = np.append(results, [np.nanmean(results, axis=0)], axis=0)
        results = np.append(results, [np.nanstd(results[:-1,:], axis=0)], axis=0)
        cases.append('Mean')
        cases.append('Std')

        print()
        print('Mean +/- Std')

        # print(f'Before: \t TRE {results[N,measures.index("tre_before")]:.3f} +/- {results[N+1,measures.index("tre_before")]:.3f}')
        # print(f'After: \t TRE {results[N,measures.index("tre")]:.3f} +/- {results[N+1,measures.index("tre")]:.3f}')
        # print()

        # print(f'Before: \t TRE {results[N,measures.index("tre_before")]:.3f} +/- {results[N+1,measures.index("tre_before")]:.3f}\t Dice {results[N,measures.index("dice_before")]:.3f} +/- {results[N+1,measures.index("dice_before")]:.3f}\t HD95 {results[N,measures.index("hd95_before")]:.3f} +/- {results[N+1,measures.index("hd95_before")]:.3f}')
        # print(f'After: \t TRE {results[N,measures.index("tre")]:.3f} +/- {results[N+1,measures.index("tre")]:.3f}\t Dice {results[N,measures.index("dice")]:.3f} +/- {results[N+1,measures.index("dice")]:.3f}\t HD95 {results[N,measures.index("hd95")]:.3f} +/- {results[N+1,measures.index("hd95")]:.3f}')
        # print()

        summary_before = 'Before: \t'
        summary_before += f'TRE {results[N,measures.index("tre_before")]:.3f} +/- {results[N+1,measures.index("tre_before")]:.3f}\t' if use_tre else ''
        summary_before += f'Dice {results[N,measures.index("dice_before")]:.3f} +/- {results[N+1,measures.index("dice_before")]:.3f}\t' if use_dice else ''
        summary_before += f'HD95 {results[N,measures.index("hd95_before")]:.3f} +/- {results[N+1,measures.index("hd95_before")]:.3f}' if use_hd95 else ''

        summary_after = 'After: \t\t'
        summary_after += f'TRE {results[N,measures.index("tre")]:.3f} +/- {results[N+1,measures.index("tre")]:.3f}\t' if use_tre else ''
        summary_after += f'Dice {results[N,measures.index("dice")]:.3f} +/- {results[N+1,measures.index("dice")]:.3f}\t' if use_dice else ''
        summary_after += f'HD95 {results[N,measures.index("hd95")]:.3f} +/- {results[N+1,measures.index("hd95")]:.3f}\t' if use_hd95 else ''
        summary_after += f'SDLogJ {results[N,measures.index("sdlogj")]:.3f} +/- {results[N+1,measures.index("sdlogj")]:.3f}\t'
        summary_after += f'MinJDet {results[N,measures.index("minJDet")]:.3f} +/- {results[N+1,measures.index("minJDet")]:.3f}\t'
        summary_after += f'MaxJDet {results[N,measures.index("maxJDet")]:.3f} +/- {results[N+1,measures.index("maxJDet")]:.3f}\t'
        summary_after += f'%NegJDet {results[N,measures.index("negJDet%")]:.3f} +/- {results[N+1,measures.index("negJDet%")]:.3f}\t'
        summary_after += f'#foldings {results[N,measures.index("num_foldings")]:.3f} +/- {results[N+1,measures.index("num_foldings")]:.3f}'

        print(summary_before)
        print(summary_after)
        print()

        if len(labels)>1:
            for m in measures_orig:
                if results_labels[m] is not None:
                    results_labels[m] = np.append(results_labels[m], [np.nanmean(results_labels[m], axis=0)], axis=0)
                    results_labels[m] = np.append(results_labels[m], [np.nanstd(results_labels[m][:-1,:], axis=0)], axis=0)


    # out_file = f'{out_dir}/results-{data["exp_name"]}.xlsx'
    out_file = f'/vol/kallisto_ssd/users/ziv/Documents/miccai23/rebuttal/results-{data["exp_name"]}-rebuttal.xlsx'
    io.save_to_excel(cases, results, measures, 'base', out_file)

    io.save_to_excel(cases, results, measures, f'{subset}-{N}', out_file)
    for m in measures_orig:
        if results_labels[m] is not None:
            io.save_to_excel(cases, results_labels[m], measures_labels[m], f'{subset}-{N}-{m}', out_file)
