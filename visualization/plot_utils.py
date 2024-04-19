import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd


def display_imagepairs_landmarks(image1, landmarks1, landmarks2, 
                                     landmarks3=None, slices=None, project=False,
                                     fig_size=(1200,800), show=True, pfile=None):
   
    image_size = image1.shape
    if slices is None:
        slices = [int(image_size[0]/2), int(image_size[1]/2), int(image_size[2]/2)]

    if landmarks3 is None:
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f'Slice [{slices[0]},:,:]',f'Slice [:,{slices[1]},:]',f'Slice [:,:,{slices[2]}]'])
    else:
        fig = make_subplots(rows=2, cols=3, subplot_titles=['Moving (red) - Fixed (blue)', 'Moving (red) - Fixed (blue)', 'Moving (red) - Fixed (blue)', 
                                                        'Moving (red) - Transformed (green)', 'Moving (red) - Transformed (green)', 'Moving (red) - Transformed (green)'])
    
    # Image 1
    fig_img1 = px.imshow(image1[:,:,slices[2]], color_continuous_scale='gray', aspect='equal')
    fig_img2 = px.imshow(image1[:,slices[1],:], color_continuous_scale='gray', aspect='equal')
    fig_img3 = px.imshow(image1[slices[0],:,:], color_continuous_scale='gray', aspect='equal')
    fig.add_trace(fig_img1.data[0], row=1, col=1)
    fig.add_trace(fig_img2.data[0], row=1, col=2)
    fig.add_trace(fig_img3.data[0], row=1, col=3)

    if landmarks3 is not None:
        # Image 1
        fig_img7 = px.imshow(image1[:,:,slices[2]], color_continuous_scale='gray', aspect='equal')
        fig_img8 = px.imshow(image1[:,slices[1],:], color_continuous_scale='gray', aspect='equal')
        fig_img9 = px.imshow(image1[slices[0],:,:], color_continuous_scale='gray', aspect='equal')
        fig.add_trace(fig_img7.data[0], row=2, col=1)
        fig.add_trace(fig_img8.data[0], row=2, col=2)
        fig.add_trace(fig_img9.data[0], row=2, col=3)

    lms_shape = landmarks1.shape 
    lms = lms_shape[0]

    df1 = pd.DataFrame(landmarks1, columns = ['X','Y','Z'])
    df2 = pd.DataFrame(landmarks2, columns = ['X','Y','Z'])
    if landmarks3 is not None:
            df3 = pd.DataFrame(landmarks3, columns = ['X','Y','Z'])
            df3.insert(0, 'Count', [str(y) for y in range(0, len(df3))])

    df1.insert(0, 'Count', [str(y) for y in range(0, len(df1))])
    df2.insert(0, 'Count', [str(y) for y in range(0, len(df2))])

    if not project:
        df1X = df1[df1['X']==slices[0]]
        df1Y = df1[df1['Y']==slices[1]]
        df1Z = df1[df1['Z']==slices[2]]
        df2X = df2[abs(df2['X']-slices[0])<=0.5]
        df2Y = df2[abs(df2['Y']-slices[1])<=0.5]
        df2Z = df2[abs(df2['Z']-slices[2])<=0.5]

        fig.add_trace(go.Scatter(x=df2X['Y'], y=df2X['Z'],mode='markers+text', text=df2X['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2Y['X'], y=df2Y['Z'],mode='markers+text', text=df2Y['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=2)
        fig.add_trace(go.Scatter(x=df2Z['X'], y=df2Z['Y'],mode='markers+text', text=df2Z['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=3)

        fig.add_trace(go.Scatter(x=df1X['Y'], y=df1X['Z'],mode='markers+text', text=df1X['Count'], marker=dict(color='blue'), textfont=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1Y['X'], y=df1Y['Z'],mode='markers+text', text=df1Y['Count'], marker=dict(color='blue'), textfont=dict(color="blue")), row=1, col=2)
        fig.add_trace(go.Scatter(x=df1Z['X'], y=df1Z['Y'],mode='markers+text', text=df1Z['Count'], marker=dict(color='blue'), textfont=dict(color="blue")), row=1, col=3)

        if landmarks3 is not None:

            df3X = df3[abs(df3['X']-slices[0])<=0.5]
            df3Y = df3[abs(df3['Y']-slices[1])<=0.5]
            df3Z = df3[abs(df3['Z']-slices[2])<=0.5]
            fig.add_trace(go.Scatter(x=df2X['Y'], y=df2X['Z'],mode='markers+text', text=df2X['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df2Y['X'], y=df2Y['Z'],mode='markers+text', text=df2Y['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=2)
            fig.add_trace(go.Scatter(x=df2Z['X'], y=df2Z['Y'],mode='markers+text', text=df2Z['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=3)

            fig.add_trace(go.Scatter(x=df3X['Y'], y=df3X['Z'],mode='markers+text', text=df3X['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df3Y['X'], y=df3Y['Z'],mode='markers+text', text=df3Y['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=2)
            fig.add_trace(go.Scatter(x=df3Z['X'], y=df3Z['Y'],mode='markers+text', text=df3Z['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=3)

    else:
        print()

        fig.add_trace(go.Scatter(x=df2['Y'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df2['X'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=2)
        fig.add_trace(go.Scatter(x=df2['X'], y=df2['Y'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=1, col=3)

        fig.add_trace(go.Scatter(x=df1['Y'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1['X'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=1, col=2)
        fig.add_trace(go.Scatter(x=df1['X'], y=df1['Y'],mode='markers+text', text=df1['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=1, col=3)

        if landmarks3 is not None:

            fig.add_trace(go.Scatter(x=df2['Y'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df2['X'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=2)
            fig.add_trace(go.Scatter(x=df2['X'], y=df2['Y'],mode='markers+text', text=df2['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=3)

            fig.add_trace(go.Scatter(x=df3['Y'], y=df3['Z'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df3['X'], y=df3['Z'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=2)
            fig.add_trace(go.Scatter(x=df3['X'], y=df3['Y'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=2, col=3)

    
    layout = fig_img1.layout
    fig.layout.coloraxis = layout.coloraxis
    fig.update_layout(width=fig_size[0], height=fig_size[1])

    # fig.show()

    if pfile is not None:
        fig.write_image(pfile)
    if show:
        fig.show()


def display_field_3d(field, factor=80,slices=None, pfile=None):

    f, ax = plt.subplots(1, 3, figsize=(20, 6))
    field_size = field.shape
    field = np.transpose(field)

    if slices is None:
        slices = [int(field_size[0]/2), int(field_size[1]/2), int(field_size[2]/2)]

    for a in range(0,3):
        axes = [0, 1, 2]
        axes.remove(a)
        # z = 64
        z = slices[a]

        fieldAx = field[axes,...].take(z, axis=a+1)

        plot_warped_grid(ax[a], factor*fieldAx, None, interval=3,title=f"axis {a}", fontsize=20)

    if pfile is not None:
        f.savefig(pfile)
        

def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="Deformation", fontsize=30, color='c'):
    """disp shape (2, H, W)
    
      source: https://github.com/qiuhuaqi/midir
    """
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    # ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_frame_on(False)

# def display_imagepairs_landmarks_ply(image1, image2, landmarks1, landmarks2, 
#                                      landmarks3=None, slices=None, project=False,
#                                      fig_size=(1200,800), show=True, pfile=None):
   
#     image_size = image1.shape
#     print(image_size)
#     if slices is None:
#         slices = [int(image_size[0]/2), int(image_size[1]/2), int(image_size[2]/2)]

#     if landmarks3 is None:
#         fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Slice [{slices[0]},:,:]',f'Slice [:,{slices[1]},:]',f'Slice [:,:,{slices[2]}]'])
#     else:
#         fig = make_subplots(rows=3, cols=3, subplot_titles=['Fixed - fLMs (red)', 'Fixed - fLMs (red)', 'Fixed - fLMs (red)', 
#                                                         'Moving - fLMs(red) + mLMs (blue)', 'Moving - fLMs(red) + mLMs (blue)', 'Moving - fLMs(red) + mLMs (blue)',
#                                                         'Fixed - fLMs(red) + regmLMs (green)', 'Fixed - fLMs(red) + regmLMs (green)', 'Fixed - fLMs(red) + regmLMs (green)'])
    
#     # Image 1
#     fig_img1 = px.imshow(image1[:,:,slices[2]], color_continuous_scale='gray', aspect='equal')
#     fig_img2 = px.imshow(image1[:,slices[1],:], color_continuous_scale='gray', aspect='equal')
#     fig_img3 = px.imshow(image1[slices[0],:,:], color_continuous_scale='gray', aspect='equal')
#     fig.add_trace(fig_img1.data[0], row=1, col=1)
#     fig.add_trace(fig_img2.data[0], row=1, col=2)
#     fig.add_trace(fig_img3.data[0], row=1, col=3)

#     # Image 2
#     fig_img4 = px.imshow(image2[:,:,slices[2]], color_continuous_scale='gray', aspect='equal')
#     fig_img5 = px.imshow(image2[:,slices[1],:], color_continuous_scale='gray', aspect='equal')
#     fig_img6 = px.imshow(image2[slices[0],:,:], color_continuous_scale='gray', aspect='equal')
#     fig.add_trace(fig_img4.data[0], row=2, col=1)
#     fig.add_trace(fig_img5.data[0], row=2, col=2)
#     fig.add_trace(fig_img6.data[0], row=2, col=3)

#     if landmarks3 is not None:
#         # Image 1
#         fig_img7 = px.imshow(image1[:,:,slices[2]], color_continuous_scale='gray', aspect='equal')
#         fig_img8 = px.imshow(image1[:,slices[1],:], color_continuous_scale='gray', aspect='equal')
#         fig_img9 = px.imshow(image1[slices[0],:,:], color_continuous_scale='gray', aspect='equal')
#         fig.add_trace(fig_img7.data[0], row=3, col=1)
#         fig.add_trace(fig_img8.data[0], row=3, col=2)
#         fig.add_trace(fig_img9.data[0], row=3, col=3)

#     lms_shape = landmarks1.shape 
#     lms = lms_shape[0]

#     df1 = pd.DataFrame(landmarks1, columns = ['X','Y','Z'])
#     df2 = pd.DataFrame(landmarks2, columns = ['X','Y','Z'])
#     # df1 = pd.DataFrame(landmarks1, columns = ['Z','X','Y'])
#     # df2 = pd.DataFrame(landmarks2, columns = ['Z','X','Y'])
#     if landmarks3 is not None:
#             df3 = pd.DataFrame(landmarks3, columns = ['X','Y','Z'])
#             # df3 = pd.DataFrame(landmarks3, columns = ['Z','X','Y'])
#             df3.insert(0, 'Count', [str(y) for y in range(0, len(df3))])

#     df1.insert(0, 'Count', [str(y) for y in range(0, len(df1))])
#     df2.insert(0, 'Count', [str(y) for y in range(0, len(df2))])

#     if not project:
#         df1X = df1[df1['X']==slices[0]]
#         df1Y = df1[df1['Y']==slices[1]]
#         df1Z = df1[df1['Z']==slices[2]]
#         df2X = df2[abs(df2['X']-slices[0])<=0.5]
#         df2Y = df2[abs(df2['Y']-slices[1])<=0.5]
#         df2Z = df2[abs(df2['Z']-slices[2])<=0.5]

#         # print(df1X)
#         # print(df1Y)
#         # print(df1Z)


#         # landmark scatter
#         # fig.add_trace(go.Scatter(x=df1X['X'], y=df1X['Y'],mode='markers+text', text=df1X['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=1)
#         # fig.add_trace(go.Scatter(x=df1Y['X'], y=df1Y['Z'],mode='markers+text', text=df1Y['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=2)
#         # fig.add_trace(go.Scatter(x=df1Z['Y'], y=df1Z['Z'],mode='markers+text', text=df1Z['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=3)

#         fig.add_trace(go.Scatter(x=df1X['Y'], y=df1X['Z'],mode='markers+text', text=df1X['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=1)
#         fig.add_trace(go.Scatter(x=df1Y['X'], y=df1Y['Z'],mode='markers+text', text=df1Y['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=2)
#         fig.add_trace(go.Scatter(x=df1Z['X'], y=df1Z['Y'],mode='markers+text', text=df1Z['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=3)

#         fig.add_trace(go.Scatter(x=df1X['Y'], y=df1X['Z'],mode='markers+text', text=df1X['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=1)
#         fig.add_trace(go.Scatter(x=df1Y['X'], y=df1Y['Z'],mode='markers+text', text=df1Y['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=2)
#         fig.add_trace(go.Scatter(x=df1Z['X'], y=df1Z['Y'],mode='markers+text', text=df1Z['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=3)


#         fig.add_trace(go.Scatter(x=df2X['Y'], y=df2X['Z'],mode='markers+text', text=df2X['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=1)
#         fig.add_trace(go.Scatter(x=df2Y['X'], y=df2Y['Z'],mode='markers+text', text=df2Y['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=2)
#         fig.add_trace(go.Scatter(x=df2Z['X'], y=df2Z['Y'],mode='markers+text', text=df2Z['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=3)

#         if landmarks3 is not None:
#             # df3 = pd.DataFrame(landmarks3, columns = ['X','Y','Z'])
#             # df3.insert(0, 'Count', [str(y) for y in range(0, len(df3))])

#             df3X = df3[abs(df3['X']-slices[0])<=0.5]
#             df3Y = df3[abs(df3['Y']-slices[1])<=0.5]
#             df3Z = df3[abs(df3['Z']-slices[2])<=0.5]
#             fig.add_trace(go.Scatter(x=df1X['Y'], y=df1X['Z'],mode='markers+text', text=df1X['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=1)
#             fig.add_trace(go.Scatter(x=df1Y['X'], y=df1Y['Z'],mode='markers+text', text=df1Y['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=2)
#             fig.add_trace(go.Scatter(x=df1Z['X'], y=df1Z['Y'],mode='markers+text', text=df1Z['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=3)

#             fig.add_trace(go.Scatter(x=df3X['Y'], y=df3X['Z'],mode='markers+text', text=df3X['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=1)
#             fig.add_trace(go.Scatter(x=df3Y['X'], y=df3Y['Z'],mode='markers+text', text=df3Y['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=2)
#             fig.add_trace(go.Scatter(x=df3Z['X'], y=df3Z['Y'],mode='markers+text', text=df3Z['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=3)

#     else:
#         print()
#         fig.add_trace(go.Scatter(x=df1['Y'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=1)
#         fig.add_trace(go.Scatter(x=df1['X'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=2)
#         fig.add_trace(go.Scatter(x=df1['X'], y=df1['Y'],mode='markers+text', text=df1['Count'], marker=dict(color='red',), textfont=dict(color="red")), row=1, col=3)

#         fig.add_trace(go.Scatter(x=df1['Y'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=1)
#         fig.add_trace(go.Scatter(x=df1['X'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=2)
#         fig.add_trace(go.Scatter(x=df1['X'], y=df1['Y'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.6), textfont=dict(color="red")), row=2, col=3)

#         fig.add_trace(go.Scatter(x=df2['Y'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=1)
#         fig.add_trace(go.Scatter(x=df2['X'], y=df2['Z'],mode='markers+text', text=df2['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=2)
#         fig.add_trace(go.Scatter(x=df2['X'], y=df2['Y'],mode='markers+text', text=df2['Count'], marker=dict(color='blue',), textfont=dict(color="blue")), row=2, col=3)

#         if landmarks3 is not None:

#             fig.add_trace(go.Scatter(x=df1['Y'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=1)
#             fig.add_trace(go.Scatter(x=df1['X'], y=df1['Z'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=2)
#             fig.add_trace(go.Scatter(x=df1['X'], y=df1['Y'],mode='markers+text', text=df1['Count'], marker=dict(color='red', opacity=0.7), textfont=dict(color="red")), row=3, col=3)

#             fig.add_trace(go.Scatter(x=df3['Y'], y=df3['Z'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=1)
#             fig.add_trace(go.Scatter(x=df3['X'], y=df3['Z'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=2)
#             fig.add_trace(go.Scatter(x=df3['X'], y=df3['Y'],mode='markers+text', text=df3['Count'], marker=dict(color='green',), textfont=dict(color="green")), row=3, col=3)

    
#     layout = fig_img1.layout
#     fig.layout.coloraxis = layout.coloraxis
#     fig.update_layout(width=fig_size[0], height=fig_size[1])

#     # fig.show()

#     if pfile is not None:
#         fig.write_image(pfile)
#     if show:
#         fig.show()
    # else:
    #     fig.close(f)

    # plot_utils.display_imagepairs_landmarks_ply(img_insp,img_exp,
    #                                     landmarks_insp[:,::-1], landmarks_exp[:,::-1],
    #                                     landmarks3 = new_landmarks_orig[:,::-1],
    #                                     slices = None, project=False, fig_size=[1000, 800])


# def display_imagepairs_landmarks(image1, image2, landmarks1, landmarks2, landmarks3=None, slices=None, project=False):
#     # image1 = fixed
#     # image2 = moving
#     # landmarks1 = fixed_landmarks
#     # landmarks2 = moving_landmarks
#     # landmarks2 = new_landmarks_orig
#     image_size = image1.shape
#     print(image_size)
#     if slices is None:
#         slices = [int(image_size[0]/2), int(image_size[1]/2), int(image_size[2]/2)]

#     f, a = plt.subplots(2, 3, figsize=(15,10))
    
#     # slices = [200,100,100]
#     print(slices)
#     a[0,0].imshow(image1[:,:,slices[2]], cmap='gray')
#     a[0,1].imshow(image1[:,slices[1],:], cmap='gray')
#     a[0,2].imshow(image1[slices[0],:,:], cmap='gray')
#     a[1,0].imshow(image2[:,:,slices[2]], cmap='gray')
#     a[1,1].imshow(image2[:,slices[1],:], cmap='gray')
#     a[1,2].imshow(image2[slices[0],:,:], cmap='gray')

#     a[0,0].set_title(f'Slice :,:,{slices[2]}')
#     a[0,1].set_title(f'Slice :,{slices[1]},:')
#     a[0,2].set_title(f'Slice {slices[0]},:,:')
#     a[1,0].set_title(f'Slice :,:,{slices[2]}')
#     a[1,1].set_title(f'Slice :,{slices[1]},:')
#     a[1,2].set_title(f'Slice {slices[0]},:,:')

#     lms_shape = landmarks1.shape 
#     lms = lms_shape[0]

#     if not project:

#         landmarks1X = landmarks1[np.where(landmarks1[:,0]==slices[0])]
#         landmarks1Y = landmarks1[np.where(landmarks1[:,1]==slices[1])]
#         landmarks1Z = landmarks1[np.where(landmarks1[:,2]==slices[2])]
#         landmarks2X = landmarks2[np.where(abs(landmarks2[:,0]-slices[0])<=0.5)]
#         landmarks2Y = landmarks2[np.where(abs(landmarks2[:,1]-slices[1])<=0.5)]
#         landmarks2Z = landmarks2[np.where(abs(landmarks2[:,2]-slices[2])<=0.5)]

        
#         a[0,0].scatter(landmarks1X[:lms,1], landmarks1X[:lms,2], color='r')
#         a[0,1].scatter(landmarks1Y[:lms,0], landmarks1Y[:lms,2], color='r')
#         a[0,2].scatter(landmarks1Z[:lms,0], landmarks1Z[:lms,1], color='r')
        

#         if landmarks3 is not None:

#             landmarks3X = landmarks3[np.where(abs(landmarks3[:,0]-slices[0])<=0.5)]
#             landmarks3Y = landmarks3[np.where(abs(landmarks3[:,1]-slices[1])<=0.5)]
#             landmarks3Z = landmarks3[np.where(abs(landmarks3[:,2]-slices[2])<=0.5)]

#             a[1,0].scatter(landmarks2X[:lms,1], landmarks2X[:lms,2], color='r', alpha=0.5)
#             a[1,1].scatter(landmarks2Y[:lms,0], landmarks2Y[:lms,2], color='r', alpha=0.5)
#             a[1,2].scatter(landmarks2Z[:lms,0], landmarks2Z[:lms,1], color='r', alpha=0.5)

#             a[1,0].scatter(landmarks3X[:lms,1], landmarks3X[:lms,2], color='b')
#             a[1,1].scatter(landmarks3Y[:lms,0], landmarks3Y[:lms,2], color='b')
#             a[1,2].scatter(landmarks3Z[:lms,0], landmarks3Z[:lms,1], color='b')
#         else:
#             a[1,0].scatter(landmarks2X[:lms,1], landmarks2X[:lms,2], color='r')
#             a[1,1].scatter(landmarks2Y[:lms,0], landmarks2Y[:lms,2], color='r')
#             a[1,2].scatter(landmarks2Z[:lms,0], landmarks2Z[:lms,1], color='r')

#     else:
#         a[0,0].scatter(landmarks1[:lms,1], landmarks1[:lms,2], color='r')
#         a[0,1].scatter(landmarks1[:lms,0], landmarks1[:lms,2], color='r')
#         a[0,2].scatter(landmarks1[:lms,0], landmarks1[:lms,1], color='r')
#         a[1,0].scatter(landmarks2[:lms,1], landmarks2[:lms,2], color='r')
#         a[1,1].scatter(landmarks2[:lms,0], landmarks2[:lms,2], color='r')
#         a[1,2].scatter(landmarks2[:lms,0], landmarks2[:lms,1], color='r')

#     plt.show()


