clear;
output_filefolder=fullfile('D:\jupyter\SRN-Deblur-master\training_set\batch_1');
dirOutput=dir(fullfile(output_filefolder,'*.jpg'));
output_filenames(:,1)={dirOutput.name};
clear dirOutput
for i=1:length(output_filenames)
    output_filenames(i)=strcat(output_filefolder(43:end),'/',output_filenames(i));
end

input_filefolder=fullfile('D:\jupyter\SRN-Deblur-master\training_set\batch_1_new');
dirOutput=dir(fullfile(input_filefolder,'*.jpg'));
input_filenames(:,1)={dirOutput.name};
clear dirOutput
for i=1:length(input_filenames)
    input_filenames(i)=strcat(input_filefolder(43:end),'/',input_filenames(i));
end

for k=2:7
    output_filefolder=fullfile(strcat('D:\jupyter\SRN-Deblur-master\training_set\batch_',num2str(k)));
    dirOutput=dir(fullfile(output_filefolder,'*.jpg'));
    output_temp(:,1)={dirOutput.name};
    clear dirOutput
    for i=1:length(output_temp)
        output_temp(i)=strcat(output_filefolder(43:end),'/',output_temp(i));
    end
    output_filenames=[output_filenames;output_temp];
    clear output_temp

    input_filefolder=fullfile(strcat('D:\jupyter\SRN-Deblur-master\training_set\batch_',num2str(k),'_new'));
    dirOutput=dir(fullfile(input_filefolder,'*.jpg'));
    input_temp(:,1)={dirOutput.name};
    clear dirOutput
    for i=1:length(input_temp)
        input_temp(i)=strcat(input_filefolder(43:end),'/',input_temp(i));
    end
    input_filenames=[input_filenames;input_temp];
    clear input_temp
end
data=[output_filenames,input_filenames]
fid = fopen('datalist.txt','w');
for i = 1:size(data, 1)
    fprintf(fid, '%s %s\n', data{i,1},data{i,2});
end
fclose(fid);