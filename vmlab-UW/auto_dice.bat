@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION
::data path
set data=D:\comp30\RawData_hist\origin\img00
set data05=D:\comp30\RawData_hist\hist0_5big\img00
set data25=D:\comp30\RawData_hist\hist0_25big\img00
set orgdata=D:\comp30\RawData\Training\label\label00

set logdir=D:\comp30\Script\log_25_weightloss_2
set logdir2=D:\comp30\Script\log_05b_weightloss_again
md %logdir%\result
md %logdir2%\result
::file name
set ct=.nii\ct.mha
set cthist05=.nii\hist_05_b.mha
set cthist25=.nii\hist_ct.mha
set label=.nii\label_8.mha

set label_mask=.nii\mask_8.mha
set weight1=%logdir%\model\model_28_0.52.hdf5
set weight2=%logdir2%\model\model_29_18.45.hdf5

::patch size
set psize=36x36x28



set numArr=26,27,28,29,30,31,32,33,34,35,36,37,38,39,40

for %%i in (%numArr%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data25%%%i%cthist25%
    set labpath=%orgdata%%%i%label%
    set result1_path=%logdir%\result\result%%i.mha
    
    set mask_path=%orgdata%%%i%label_mask%
    set pinfo=.\info\ID%%iinfo.txt

    segmentation3DUnet.py !cthistpath! D:\comp30\Script\444428c9.yml %weight1% !result1_path!
)

goto start
set numArr2=26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
for %%i in (%numArr2%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data2%%%i%cthist%
    set labpath=%orgdata%%%i%label%
    set result2_path=%logdir2%\result\result%%i.mha
    
    set mask_path=%orgdata%%%i%label_mask%
    set pinfo=.\info\ID%%iinfo.txt

    segmentation3DUnet.py !cthistpath! D:\comp30\Script\444428c9.yml %weight2% !result2_path!
)
:start
::set numArr=01,02,03,04,05,06,07,08,09,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
