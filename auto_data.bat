@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION
::data path
set data=D:\comp30\RawData_hist\hist0_10f\img00
set orgdata=D:\comp30\RawData\Training\label\label00
set toolbin=C:\study\bin\imageproc\bin\imageproc.exe
::file name
set ct=.nii\ct.mha
set cthist=.nii\hist_10_f.mha
set label=.nii\label_8.mha
set label1=\label_Liver.mha
set label2=\label_PortalVein.mha
set label3=\label_HepaticVein.mha
set label4=\label_vessel.mha
set label5=\
set withMask=\new\ct_withmask.mha
set label_result1=\new\label_result_withMask1e3.mha
set label_result2=\new\label_result_withMask1e4.mha
set label_mask=.nii\mask_8.mha
set weight1=D:\comp30\Script\log_mask8\latestweights.hdf5
set weight2=D:\comp30\Script\log_mask8\bestweights.hdf5

::patch size
set psize=36x36x28


:: NO 15 and 25
set numArr=01,02,03,04,05,06,07,08,09,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40

for %%i in (%numArr%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data%%%i%cthist%
    set labpath=%orgdata%%%i%label%
    set l1path=%data%%%i%label1%
    set l2path=%data%%%i%label2%
    set l3path=%data%%%i%label3%
    set l4path=%data%%%i%label4%
    set l5path=%data%%%i%label5%
    set result1_path=D:\comp30\Script\log_mask8\result\result%%i.mha
    set result2_path=%data%%%i%label_result2%
    set mask_path=%orgdata%%%i%label_mask%
    set pinfo=.\info\ID%%iinfo.txt

    
    extractpatch.py !cthistpath! 444428.yml patch_mask8\hist0_10f\hist%%i list\mask8\hist0_10f\histID%%i.txt --mask !mask_path!
)


    ::segmentation3DUnet.py !ctpath! C:\study\vessel\3DUnet\onetype.yml %weight1% !result1_path! --mask !mask_path!
    ::%toolbin% load:!nl4path! similarity:!result1_path!,dice >>log_two_trans\result_L.txt
    ::segmentation3DUnet.py !ctpath! C:\study\vessel\3DUnet\onetype.yml %weight2% !result2_path! --mask !mask_path!
    ::%toolbin% load:!nl4path! similarity:!result2_path!,dice >>log_two_trans\result_B.txt
     ::ExtractPatchImages.exe !pinfo! --input !nl1path! !nl2path! !nl3path! !nl4path! --outdir patch\4class_label%%i --patchlist .\list\list_4class\lID%%i.txt -t 8 --compose --withBG
     ::ExtractPatchImages.exe !pinfo! --input !l4path! --outdir patch\v_label%%i --patchlist .\list\vID%%i.txt -t 8 --compose --withBG
    ::ExtractPatchImages.exe !pinfo! --input !ctpath! --outdir patch\origin%%i --patchlist .\list\oID%%i.txt -t 8
    ::extractpatch.py !ctpath! onetype.yml patch\origin%%i list\oID%%i.txt --mask !mask_path!
    ::extractpatch_composed.py !l4path! onetype.yml patch\vlabel%%i list\vlID%%i.txt --mask !mask_path!
::    extractpatch_composed.py !labpath! 444428.yml patch2\label%%i list\labelID%%i.txt
::    extractpatch.py !cthistpath! 444428.yml patch\hist%%i list\histID%%i.txt
::    extractpatch.py !cthistpath! 444428.yml patch_mask8\hist%%i list\mask8\histID%%i.txt --mask !mask_path!
::segmentation3DUnet.py !ctpath! C:\study\vessel\3DUnet\onetype.yml %weight2% !result2_path!
::extractpatch.py !cthistpath! 444428.yml patch_mask8\hist1_0\hist%%i list\mask8\hist1_0\histID%%i.txt --mask !mask_path!
::set numArr=01,02,03,04,05,06,07,08,09,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
::segmentation3DUnet.py !ctpath! D:\comp30\Script\444428c9.yml %weight1% !result1_path!