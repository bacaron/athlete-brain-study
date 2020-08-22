function [] = produceAggregateNetworkImages(topDir,groupings,measures,ranges,saveDir)

for gg = 1:length(groupings)
    for mm = 1:length(measures)
        data = load(fullfile(topDir,groupings{gg},sprintf('%s.mean.csv',measures{mm})));
        colormap hot;
        if strcmp(measures{mm},'count') || strcmp(measures{mm},'density')
            imagesc(log10(data),ranges{mm});
        else
            imagesc(data,ranges{mm});
        end
        colorbar

        %axis square

        set(gcf,'Position',[0,0,600,600])
        
        saveas(gcf,fullfile(saveDir,sprintf('%s_%s_mean.jpg',measures{mm},groupings{gg})),'jpg');
        close gcf
    end
end