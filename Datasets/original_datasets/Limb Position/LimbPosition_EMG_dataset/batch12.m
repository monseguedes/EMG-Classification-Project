function batch12(Subject)
position = {'Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5'};
%%
inFileName = 'names1.txt';
for jj = 1:numel(position)
    EMG_file = GetEMGFileNames (inFileName);
    DATA=[];
    for fn =1:size(EMG_file,1) %For each file name string
        category = getCategory(EMG_file{fn});
        fname = ['Pos' num2str(jj) '_' EMG_file{fn}];
        [x] = textread(fname,'','delimiter',' ');
        wininc = 100;
        winsize = 400;
        x=x(:,1:7);
        D=[];
        feature1 = getYOUROWNfeat(x,winsize,wininc);
        D = [D  feature1];
        D =[D repmat(category,size(feature1,1),1) ];
        DATA=[DATA; D];
    end
    outFileName = [Subject '_trial1_' 'Pos' num2str(jj) '_' 'data'];
    save(outFileName, 'DATA')
end