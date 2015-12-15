function combinedBatchLogFile = combineBatchLogs(batch_folder,batch_range)

    if numel(batch_range)==1
        batch_range = [batch_range, batch_range];
    end

    combinedBatchLogFile = [batch_folder,filesep,'data_batches',...
        num2str(batch_range(1)),'-',num2str(batch_range(2)),'.txt'];
    
    if exist(combinedBatchLogFile,'file')
        dos(['del ',combinedBatchLogFile])
        if exist(combinedBatchLogFile,'file')
            error('deletion not always working because of permission issues for file: %s',combinedBatchLogFile)
        end
    end
    
    for batch = batch_range(1):batch_range(2)
        fprintf(' adding batch log %d of %d...\n', batch,batch_range(2)-batch_range(1)+1)
        curr_batch = rdir([batch_folder,filesep,'data_batch_',num2str(batch),'.txt']);
        if ~isempty(curr_batch)
            
            command = ['type ',curr_batch.name,' 1>>',combinedBatchLogFile];
            [status, result] = dos(command,'-echo');
            if status~=0
                error(result)
            end
        else
            error(' Could not find batch %!',batch)
        end
    end
    