clear;

%Jpeg directory: save converted dicoms 
JpgDir=[pwd '\datasets\ColitisDevkit2015\Colitis2015\'];

%Dicom directory
DicomDir='data\Colitis\';  %directory of dicoms

%Patient Name
patientList={'302','304','306'};


for kk=1:length(patientList)
    
    patient=patientList{kk};    
    
    %load dicom files
    info=dicom_read_header([DicomDir patient]);
    Et = dicom_read_volume(info);
    
    %edit here to set slices to convert
    slice_start=round(size(Et,3)/2);
    slice_end=size(Et,3)-20;
    
    %intensity region for colitis (HU+1024)
    intensity_low=890;
    intensity_high=1240;
        
    for zz=slice_start:slice_end
       
        R = Et(:,:,zz);
        
        if info.RescaleIntercept==0
            R1=R+1024;
        end
        if info.RescaleIntercept==-1024
            R1=R;
        end
        
        %line scale CT values to [0 255]
        R1=Et(:,:,zz);
        R1(R1>=intensity_high)=intensity_high;
        R1(R1<=intensity_low)=intensity_low;
        R8=255*double(R1-intensity_low)/double(intensity_high-intensity_low);
        R8=uint8(R8);
        G8=R8;
        B8=R8;
                  
        
        %generate 3 channel jpg 
        I=cat(3,uint8(G8),uint8(G8),uint8(G8));
        imwrite(I,[JpgDir patient '_' sprintf('%04d',zz) '.jpg']);
     
 
    end
   
end

