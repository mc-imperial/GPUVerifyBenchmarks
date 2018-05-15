//pass
//--blockDim=256 --gridDim=2

#include "common.h"

///////////////////////////////////////
//// Compute forward substring matches
///////////////////////////////////////

__global__ void
mummergpuKernel(MatchCoord* match_coords,
             char* queries, 
             const int* queryAddrs,
			 const int* queryLengths,
             const int numQueries,
			 const int min_match_len) 
{
   int qryid = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (qryid >= numQueries) { return; }
   XPRINTF("> qryid: %d\n", qryid);

   int qlen = queryLengths[qryid];

   // start at root for first query character
   TextureAddress cur;
   cur.data = 0;

   int mustmatch = 0;
   int qry_match_len = 0;

   int qryAddr=queryAddrs[qryid];
   MatchCoord * result = match_coords + qryAddr - __umul24(qryid, min_match_len + 1);
   queries += qryAddr;

   int last = qlen - min_match_len;

   for (int qrystart = 0;
       qrystart <= last;
       qrystart++, result++, queries++)
   {
	  XPRINTF("qry: %s\n", queries + 1);

	  PixelOfNode node;
      TextureAddress prev;

      if ((cur.data == 0) || (qry_match_len < 1))
      {
	    // start at root of tree
	    cur.x = 0; cur.y = 1;
	    qry_match_len = 1; 
        mustmatch = 0;
      }

	  char c = queries[qry_match_len];

	  XPRINTF("In node (%d,%d): starting with %c [%d] =>  \n", cur.x, cur.y, c, qry_match_len);

	  int refpos = 0;
	  while ((c != '\0'))
	  {
		 XPRINTF("Next edge to follow: %c (%d)\n", c, qry_match_len);

	     PixelOfChildren children;
		 children.data = tex2D(childrentex,cur.x, cur.y);
		 prev = cur;

		 switch(c)
		 {
			case 'A': cur=children.children[0]; break;
			case 'C': cur=children.children[1]; break;
			case 'G': cur=children.children[2]; break;
			case 'T': cur=children.children[3]; break;
            default: cur.data = 0; break;
		 };		 

		 XPRINTF(" In node: (%d,%d)\n", cur.x, cur.y);

		 // No edge to follow out of the node
         if (cur.data == 0)
		 {
			XPRINTF(" no edge\n");
	        set_result(prev, result, 0, qry_match_len, min_match_len, 
                       FORWARD); 

            qry_match_len -= 1;
            mustmatch = 0;

			goto NEXT_SUBSTRING;
		 }

         {
           unsigned short xval = cur.data & 0xFFFF;
           unsigned short yval = (cur.data & 0xFFFF0000) >> 16;
		   node.data = tex2D(nodetex, xval, yval);
         }

		 XPRINTF(" Edge coordinates: %d - %d\n", node.start, node.end);

         if (mustmatch)
         {
           int edgelen = node.end - node.start+1;
           if (mustmatch >= edgelen)
           {
             XPRINTF(" mustmatch(%d) >= edgelen(%d), skipping edge\n", mustmatch, edgelen);

             refpos = node.end+1;
             qry_match_len += edgelen;
             mustmatch -= edgelen;
           }
           else
           {
             XPRINTF(" mustmatch(%d) < edgelen(%d), skipping to:%d\n", 
                     mustmatch, edgelen, node.start+mustmatch);

             qry_match_len += mustmatch;
             refpos = node.start + mustmatch;
             mustmatch = 0;
           }
         }
         else
         {
           // Try to walk the edge, the first char definitely matches
           qry_match_len++;
           refpos = node.start+1;
         }

		 c = queries[qry_match_len];

		 while (refpos <= node.end && c != '\0')
		 { 
            char r = getRef(refpos);

			XPRINTF(" Edge cmp ref: %d %c, qry: %d %c\n", refpos, r, qry_match_len, c);
						
			if (r != c)
			{
			   // mismatch on edge
			   XPRINTF("mismatch on edge: %d, edge_pos: %d\n", qry_match_len,refpos - (node.start));
               goto RECORD_RESULT;
			}

	        qry_match_len++;
			refpos++;
			c = queries[qry_match_len];
		 }
	  }

	  XPRINTF("end of string\n");

      RECORD_RESULT:
	
      set_result(cur, result, refpos - node.start, qry_match_len, 
                 min_match_len, FORWARD);

      mustmatch = refpos - node.start;
      qry_match_len -= mustmatch + 1;

      NEXT_SUBSTRING:

      node.data = tex2D(nodetex, prev.x, prev.y);
      cur = node.suffix;

      XPRINTF(" following suffix link. mustmatch:%d qry_match_len:%d sl:(%d,%d)\n", 
              mustmatch, qry_match_len, cur.x, cur.y);

      do {} while(0);
   }
	
   return;

}
